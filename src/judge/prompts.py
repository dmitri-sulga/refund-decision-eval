"""Judge prompt versions. A version name is frozen once it has cached responses
(the cache stores a hash of the exact request and refuses silent edits)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import config as C

DECISION_KEY = "monetary_relief"


def visible_narrative(text: str) -> str:
    """The narrative exactly as every model sees it (long ones are cut)."""
    if len(text) <= C.MAX_NARRATIVE_CHARS:
        return text
    return text[: C.MAX_NARRATIVE_CHARS].rstrip() + " [narrative truncated]"


def _case_header(rec: dict) -> str:
    return (f"Product: {rec['product']} / {rec['sub_product']}\n"
            f"Issue selected by the consumer: {rec['issue']} / {rec['sub_issue']}\n")


# ====================================================================== v1
V1_SYSTEM = """You are a disputes analyst at a credit card issuer. You read a cardholder's complaint about a purchase shown on their statement and decide how it should be resolved:
- monetary_relief = true: the cardholder should get money back (a refund, a statement credit, or a reversal of the disputed charge or of related fees and interest).
- monetary_relief = false: the complaint should be closed with an explanation, or with non-monetary action only.

Rubric:
1. Unauthorized charges (the card was used without the cardholder's permission) normally qualify for relief.
2. Billing errors (duplicate charge, wrong amount, a merchant refund that was never credited) normally qualify for relief.
3. Goods or services not received, or materially not as described, qualify for relief when the cardholder tried to resolve it with the merchant and the merchant failed to fix it.
4. No relief when the cardholder (or someone they allowed) made the purchase and got what was agreed, when the complaint is only dissatisfaction with a disclosed price, policy or service, or when the problem has already been resolved.

Reply with JSON only:
- monetary_relief: true or false
- p_relief: your probability, from 0 to 1, that monetary relief is the right resolution
- reason: one sentence, at most 15 words"""

V1_SCHEMA = {
    "type": "object",
    "properties": {
        "monetary_relief": {"type": "boolean"},
        "p_relief": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["monetary_relief", "p_relief", "reason"],
}


def _v1_user(rec: dict, narrative: str) -> str:
    return (_case_header(rec) + "Complaint narrative:\n\"\"\"\n" + narrative + "\n\"\"\"")


# ====================================================================== v2
# Written after the v1 error analysis on DEV. Changes, each traced to the analysis:
# - v1 granted relief in most DEV cases although most complaints end without it:
#   state the base rate and that the narrative is one-sided (calibration).
# - v1 missed issuer-side mistakes and invented exclusions: name issuer errors as
#   a relief signal and forbid invented rules.
# - Cue directions were checked on TRAIN with keyword proxies before being
#   written down; the largest v1 error category (a prior denial by the issuer)
#   did not separate outcomes on TRAIN, so it is recorded as a fact, not a rule.
# - Facts are extracted before the decision, for the fact model and for audits.
V2_SYSTEM = """You review complaints that credit card holders filed with a regulator about a purchase shown on their statement. The card issuer has already answered each complaint. Predict how the issuer's answer was recorded:
- monetary_relief = true: closed with monetary relief (money back: a refund, a statement credit, or a reversal of a charge, fee or interest).
- monetary_relief = false: closed with an explanation, or with non-monetary action only.

What past decisions on these complaints look like:
- About 3 in 10 end with monetary relief; most end with an explanation.
- You see only the cardholder's side. Most complainants had already disputed the charge with the issuer and many had already been turned down. The issuer usually holds evidence the narrative does not show (chip or device data, delivery proof, the merchant's rebuttal, account history). A complaint that sounds convincing is not by itself a reason to predict relief.
- Relief was more common when: the disputed amount is small; the issuer itself made a specific mistake (a refund or credit it acknowledged but did not apply, a charge it cannot explain, a missed deadline, rebilling before its investigation ended); fees or interest were charged because of the disputed charge; the card was replaced after a prompt fraud report.
- Relief was less common when: no amount or no clear transaction is described; the complaint is about the quality of a service that was delivered; the narrative is short or vague.
- Use only what the narrative says. Do not invent rules, exclusions or deadlines.

First record the facts, then decide. Reply with JSON only:
- claim: "unauthorized", "not_received", "not_as_described", "billing_error", "cancellation", or "other"
- issuer_status, what the issuer had done before this complaint: "denied", "pending", "credited", "partly_credited", "not_contacted", or "unclear"
- issuer_error: true only if the narrative describes a specific, checkable mistake by the issuer itself
- ask: "money" or "non_money" (for example closing an account or fixing a credit report)
- detail: "specific" or "vague"
- monetary_relief: true or false
- p_relief: probability from 0 to 1 that the recorded answer was monetary relief"""

V2_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string", "enum": ["unauthorized", "not_received", "not_as_described",
                                             "billing_error", "cancellation", "other"]},
        "issuer_status": {"type": "string", "enum": ["denied", "pending", "credited",
                                                     "partly_credited", "not_contacted", "unclear"]},
        "issuer_error": {"type": "boolean"},
        "ask": {"type": "string", "enum": ["money", "non_money"]},
        "detail": {"type": "string", "enum": ["specific", "vague"]},
        "monetary_relief": {"type": "boolean"},
        "p_relief": {"type": "number"},
    },
    "required": ["claim", "issuer_status", "issuer_error", "ask", "detail", "monetary_relief",
                 "p_relief"],
}


# ====================================================================== registry
@dataclass(frozen=True)
class PromptVersion:
    name: str
    system: str
    schema: dict
    user: Callable[[dict, str], str]
    preprocess: Callable[[str], str] | None = None   # applied to the narrative text


VERSIONS: dict[str, PromptVersion] = {
    "v1": PromptVersion("v1", V1_SYSTEM, V1_SCHEMA, _v1_user),
    "v2": PromptVersion("v2", V2_SYSTEM, V2_SCHEMA, _v1_user),
}


def build_messages(version: str, rec: dict, narrative: str) -> tuple[list, dict]:
    """narrative: the visible (possibly attacked) narrative text."""
    pv = VERSIONS[version]
    text = pv.preprocess(narrative) if pv.preprocess else narrative
    return ([{"role": "system", "content": pv.system},
             {"role": "user", "content": pv.user(rec, text)}], pv.schema)
