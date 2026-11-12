# Cloud branch (Amendment 3): what the independent reviewer must verify before any request

Written with the amendment, before any request of the branch. The reviewer is a separate session that did not write
this code. Every item is checked against the files and the public documentation, not against this list's wording. A
reviewer who finds a problem writes it down with the file and line; nothing is fixed silently, and any fix is made
before publication and recorded in `docs/deviations.md`.

## A. The plan and the record
1. `docs/analysis_plan_models.md`: lines 1–149, Amendment 1 and Amendment 2 are byte-identical to `60dfe9b`
   (`git diff 60dfe9b -- docs/analysis_plan_models.md` shows only added lines after the last line of Amendment 2).
   `python -m judge.models --verify` passes (it checks lines 1–149 against `162909f`).
2. Amendment 3 says, and the code does, the same thing on every point: models and order (§2 table against
   `FREE_QUEUE`, `PAID_QUEUE`, `ENDPOINT` in `src/judge/cloud.py`), comparison set and corrected levels (§3 against
   `cloud_analysis.N_A`, `N_B`), request fields (§5 against `cloud.build_body`), caps and the cost rule (§7 against
   `CAP_USD`, `SUB_CAP_USD`, `cost`, `worst_case`, `projected`), error handling (§9 against `classify`, `RETRY_DELAYS_S`,
   `MAX_ERRORS_PER_MODEL`, `first_request_drop`, `close_incomplete`, `repeated`), schedule (§10 against
   `MIN_INTERVAL_S`, `FREE_DAILY_MARGIN`, `FREE_DAILY_FALLBACK`), refusals before a run (§12 against `preflight`).
3. Nothing in the amendment states a result: no answer of any model of the branch exists
   (`results/models/cloud/` holds only `registration.json`, `pre_run_checklist.md` and `snapshot/`).
4. The twelfth-change entry in `docs/deviations.md`, the records in `results/post_run_changes.json` and the README's
   deviations paragraph say the same thing; `results/metrics.json` differs from `60dfe9b` only in the post-run change
   records (compare leaf by leaf).

## B. The model selection
5. `python -m judge.cloud_select --check` passes, and `--resolve` prints the registered list, the rejected free
   models with the reasons the amendment gives, and the next paid models.
6. The snapshot is what OpenRouter and Hugging Face returned: spot-check at least the 7 selected models and 3 rejected
   ones against the live listings (`curl https://openrouter.ai/api/v1/models`, the `/endpoints` URL of each, the
   Hugging Face API URL in `fetch_log.json`). Differences since 2026-11-12 are expected for prices and status; note
   any that would change the selection, and do not re-take the snapshot.
7. The rules are reasonable and were not tuned to a result: read `cloud_select.py`; check the one correction the
   amendment discloses (rule 6, "built on"); try the rules with each threshold moved a little (32B, 1.00 / 4.00 USD,
   fp8) and record which choices are sensitive to it.
8. Excluded on purpose: Anthropic, OpenAI, routers, mandatory-reasoning, abliterated or uncensored, coding and other
   special-purpose models, derivatives, and every model already run (local or hosted). Confirm none of the seven is any
   of these, and that `gemma-4-31b-it:free` is a reference row only.

## C. The request and the answers
9. `tests/test_cloud.py::test_the_request_carries_the_frozen_messages_and_schema_and_only_the_registered_settings`:
   the messages rebuild the frozen request hash in `results/frozen.json`; the schema is the frozen one plus
   `additionalProperties: false`, and only where the endpoint supports structured outputs.
10. The body fields and their meaning match OpenRouter's documentation of 2026-11-12: `reasoning.effort "none"`,
    `provider.order` with an endpoint tag, `allow_fallbacks: false`, `require_parameters: true`,
    `response_format` json_schema with `strict`, `seed`, `temperature`, `max_tokens`. Record any field whose effect
    the documentation does not state.
11. An answer's text, finish reason and token counts are read from the documented places
    (`choices[0].message.content`, `finish_reason`, `usage.prompt_tokens`, `usage.completion_tokens`,
    `usage.cost`, `usage.completion_tokens_details.reasoning_tokens`), and `models_analysis.malformed` treats every
    `finish_reason` other than `stop` as malformed.
12. The served-model check (`model_matches`) accepts the registered model's names and nothing else; the provider check
    compares with the endpoint's `provider_name` in the snapshot. Judge whether a legitimate answer could be rejected
    by either (that would close a model).

## D. Money and the key
13. `python -m judge.cloud --estimate` prints projected 4.6343, worst case 10.1305 and basis 2.9022 USD; recompute the
    amendment's estimate table by hand from the prices and the token basis.
14. The caps cannot be exceeded by the code's own accounting: every request is reserved at its worst case before it
    is journaled; tests `test_the_model_sub_cap_stops...` and `test_the_branch_cap_stops...` pass; a free model's
    reported cost is counted.
15. The key: read only in `read_key` when a request is sent; sent only as `Authorization` with
    `add_unredirected_header`; redirects not followed; errors scrubbed; `main` prints no traceback; nothing under
    `results/models/cloud/` matches `KEY_PATTERN`; the Makefile names neither the key nor its file. The reviewer does
    NOT open or print the real key file; at most `stat -f '%Sp' ~/.openrouter_key` (expect `-rw-------`).
16. The credit and key-limit readings (`account_readings`) are GET requests with the key that call no model; confirm
    against the documentation that they cost nothing.

## E. Held-out, order, stops
17. No held-out complaint can be sent: `test_an_injected_held_out_id_sends_journals_and_spends_nothing` passes, and
    `X.guard` is called before every journal entry in `cloud._run`.
18. The runner refuses (read `preflight`): an unclosed model of Amendment 1 or 2, an incomplete registration, a changed
    protected file, a dirty tree outside `results/models/cloud/`, an unpublished HEAD, an earlier model of the queue not
    closed, unlocked labels, a missing or group-readable key file, and a failed listing check.
19. Free-tier handling: the daily allowance stops a run before OpenRouter's limit; a 429 with
    `X-RateLimit-Remaining: 0` on a free model is a pause, not an error; a data-policy refusal is an account
    condition and cannot close a model.
20. Closing: only the conditions of §9; the 24-hour rule reads the server's `date` headers; `make check` fails if the
    journals' times go backwards.

## F. The commands (check each command's own exit status, not a pipe's: lesson L-pipe of `docs/deviations.md`)
21. `make test` exits 0 (`make test; echo $?`).
22. `make check` exits 0.
23. `git status` is clean after the commit; no file outside `docs/`, `src/judge/`, `tests/`, `Makefile`,
    `results/models/cloud/`, `results/post_run_changes.json`, `results/metrics.json`, `results/report.md`, `README.md`
    changed.

## G. What the reviewer reports
A list of findings, each with severity (blocks publication / should fix / note), file and line, and evidence. Then a
single line: "publish" or "do not publish". The owner decides; nothing is pushed, no gist is created and no request is
sent before that.
