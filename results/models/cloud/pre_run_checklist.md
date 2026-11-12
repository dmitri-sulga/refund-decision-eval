# Cloud branch (Amendment 3): what the independent reviewer must verify before any request

Written with the amendment, before any request of the branch; updated with the amendment's Correction after the first
independent review (`a192234` reviewed; its findings B1, B2, M1–M7 and m1–m11 are answered by the Correction, C1–C10).
The reviewer is a separate session that did not write this code. Every item is checked against the files and the public documentation, not against this list's wording. A
reviewer who finds a problem writes it down with the file and line; nothing is fixed silently, and any fix is made
before publication and recorded in `docs/deviations.md`.

## A. The plan and the record
1. `docs/analysis_plan_models.md`: lines 1–779 (lines 1–149, Amendment 1 and Amendment 2) are byte-identical to
   `60dfe9b`, and lines 780–1094 (Amendment 3 as first reviewed) to `a192234`; the Correction is only appended after
   them (`cmp <(git show 60dfe9b:docs/analysis_plan_models.md) <(head -n 779 docs/analysis_plan_models.md)`, the same
   with `a192234` and 1094, and `git diff a192234 -- docs/analysis_plan_models.md` has no removed line).
   `python -m judge.models --verify` passes (it checks lines 1–149 against `162909f`).
2. Amendment 3 with its Correction says, and the code does, the same thing on every point (where they differ, the
   Correction holds): models and order (§2 table against
   `FREE_QUEUE`, `PAID_QUEUE`, `ENDPOINT` in `src/judge/cloud.py`), comparison set and corrected levels (§3 against
   `cloud_analysis.N_A`, `N_B`), request fields (§5 against `cloud.build_body`), caps and the cost rule (§7 against
   `CAP_USD`, `SUB_CAP_USD`, `cost`, `worst_case`, `projected`), error handling (§9 against `classify`, `RETRY_DELAYS_S`,
   `MAX_ERRORS_PER_MODEL`, `first_request_drop`, `close_incomplete`, `repeated`), schedule (§10 against
   `MIN_INTERVAL_S`, `FREE_DAILY_MARGIN`, `FREE_DAILY_FALLBACK`), refusals before a run (§12 against `preflight`);
   and the Correction: C1 against `account_readings` and `paid_key_refusal`, C2 against `listing_check`,
   `record_listing`, `unavailable_streak`, `streak_closes`, `repeated` and `watch_listing`, C3 against
   `overrun_headroom` and the overrun branch of `_run`, C4 against `provider_matches` and the quarantine, C5 against
   `reasoning_in_content`, C6 against `cloud_analysis` (`registered_outputs`), C8 against the summary sentence.
3. Nothing in the amendment or its Correction states a result: no answer of any model of the branch exists
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
   fp8) and compare with the sensitivities the Correction states (C10: P1 at 0.99 against the 1.00 ceiling).
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
    (`provider_matches`) compares, after case folding and removing punctuation, with the snapshot's `provider_name`
    and the tag's provider part, and accepts a missing field (C4). Judge whether a legitimate answer could be rejected
    by either; a first mismatch only stops the run and quarantines the answer, a second closes the model.

## D. Money and the key
13. `python -m judge.cloud --estimate` prints projected 4.6343, worst case 10.1305 and basis 2.9022 USD; recompute the
    amendment's estimate table by hand from the prices and the token basis.
14. The caps cannot be exceeded by the code's own accounting: every request is reserved at its worst case before it
    is journaled, and the branch cap keeps room for one overrun answer (C3); tests `test_the_model_sub_cap_stops...`,
    `test_the_branch_cap_stops...`, `test_the_branch_cap_keeps_room...` and the three overrun tests pass; a free
    model's reported cost is counted; an overrun is journaled and `make check` still passes after one.
15. The key: read only in `read_key` when a request is sent; sent only as `Authorization` with
    `add_unredirected_header`; redirects not followed; errors scrubbed; `main` prints no traceback; nothing under
    `results/models/cloud/` matches `KEY_PATTERN`; the Makefile names neither the key nor its file. The reviewer does
    NOT open or print the real key file; at most `stat -f '%Sp' ~/.openrouter_key` (expect `-rw-------`).
16. The key readings (`account_readings`) are `GET /api/v1/key` only (no `/credits`, which needs a management key),
    call no model and cost nothing; the paid arm refuses a key whose `limit` is null, above 15 USD or resetting, or
    whose `limit_remaining` is short (C1). Confirm the field names against `api-reference/limits` on the day.

## E. Held-out, order, stops
17. No held-out complaint can be sent: `test_an_injected_held_out_id_sends_journals_and_spends_nothing` passes, and
    `X.guard` is called before every journal entry in `cloud._run`.
18. The runner refuses (read `preflight`; `test_preflight_refuses_each_registered_condition`): an unclosed model of
    Amendment 1 or 2, an incomplete registration, a changed protected file, a dirty tree outside
    `results/models/cloud/`, an unpublished HEAD, an earlier model of the queue not closed, a closed model, a paid model
    after the branch cap, unlocked labels, a missing or group-readable key file, and a failed listing check (not the
    endpoint's status, which is recorded: C2).
19. Free-tier handling: the daily allowance stops a run before OpenRouter's limit; a 429 with
    `X-RateLimit-Remaining: 0` on a free model is a pause, not an error, unless it is the per-minute limit (C10); a
    data-policy refusal is an account condition and cannot close a model.
20. Closing: only the conditions of §9 as corrected; the 24-hour rules read the server's `date` headers and need no
    answer (and, for the listing, no passing check) in between and hourly checks (C2); `make check` fails if the
    journals' times go backwards.

## F. The commands (check each command's own exit status, not a pipe's: lesson L-pipe of `docs/deviations.md`)
21. `make test` exits 0 (`make test; echo $?`).
22. `make check` exits 0.
23. `git status` is clean after the commit; no file outside `docs/`, `src/judge/`, `tests/`, `Makefile`,
    `results/models/cloud/`, `results/post_run_changes.json`, `results/metrics.json`, `results/report.md`, `README.md`
    changed.

## G. The start day (the author and the operator, after publication and registration, before the first request)
24. A key created for this branch only, with a credit limit of at most 15 USD that never resets, set in OpenRouter
    and saved in the key file (`--key-file`, `OPENROUTER_KEY_FILE`, or the default in the author's home directory),
    readable only by its owner (C1). Preferably in a separate OpenRouter account or workspace (C9).
25. The privacy setting for free and for paid models decided by the author (C9), and written here with its time:
    free: ______ paid: ______ at ______ UTC; account or workspace: ______.
26. `python -m judge.cloud --status` run once; its output (`limit`, `limit_reset`, `free_model_daily_requests.limit`)
    copied into `registration.json` as `key_reading` and published with the registration (C10). If the free daily
    limit is 50, not 1,000, the schedule of §10 does not hold (C10).
27. `python -m judge.cloud --check-listings` run on the start day, before the first request, to re-read the live
    listings of all seven models (C2); its records committed. A model whose check fails is not replaced; it waits, and
    `--watch-listing MODEL` decides whether it is ever closed.

## H. What the reviewer reports
A list of findings, each with severity (blocks publication / should fix / note), file and line, and evidence. Then a
single line: "publish" or "do not publish". The owner decides; nothing is pushed, no gist is created and no request is
sent before that.
