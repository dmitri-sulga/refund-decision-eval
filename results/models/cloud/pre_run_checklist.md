# Cloud branch (Amendment 3): what the independent reviewer must verify before any request

Written with the amendment, before any request of the branch; updated with the amendment's Correction after the first
independent review (`a192234` reviewed; its findings B1, B2, M1–M7 and m1–m11 are answered by the Correction, C1–C10),
and with Correction 2 after the second (`c01188a` reviewed; its findings MAJOR-1 to MAJOR-3 and m-a to m-m are answered
by Correction 2, C11–C17), and with Correction 3 after the third (`0287331` reviewed; its findings MAJOR-1 to MAJOR-3
and m-1 to m-7 are answered by Correction 3, C18–C21, which also records the documentation it was checked against on
2026-11-12), and with Correction 4 after the fourth (`959a2db` reviewed; its findings MAJOR-1, MAJOR-2 and m-1 to m-7
are answered by Correction 4, C22–C25, with the documentation read again on 2026-11-12 16:39 UTC).
The reviewer is a separate session that did not write this code. Every item is checked against the files and the public documentation, not against this list's wording. A
reviewer who finds a problem writes it down with the file and line; nothing is fixed silently, and any fix is made
before publication and recorded in `docs/deviations.md`.

## A. The plan and the record
1. `docs/analysis_plan_models.md`: lines 1–779 (lines 1–149, Amendment 1 and Amendment 2) are byte-identical to
   `60dfe9b`, lines 780–1094 (Amendment 3 as first reviewed) to `a192234`, lines 1095–1239 (the Correction as
   reviewed the second time) to `c01188a`, lines 1240–1344 (Correction 2 as reviewed the third time) to
   `0287331`, and lines 1345–1474 (Correction 3 as reviewed the fourth time) to `959a2db`; each correction is only
   appended after them
   (`cmp <(git show 60dfe9b:docs/analysis_plan_models.md) <(head -n 779 docs/analysis_plan_models.md)`, the same with
   `a192234` and 1094, with `c01188a` and 1239, with `0287331` and 1344 and with `959a2db` and 1474, and
   `git diff 959a2db -- docs/analysis_plan_models.md` has no removed line).
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
   `reasoning_in_content`, C6 against `cloud_analysis` (`registered_outputs`), C8 against the summary sentence; and
   Correction 2: C11 against `_error_from_json`, `budget_spent` and `classify`, C12 against `filtered_counts`, the
   `filtered` branch and the end of `_run`, and the content-filter check of `verify`, C13 against `key_limit_stop`,
   `arm_stopped_by_key`, `preflight` and `close_incomplete`, C14 against `note_overrun`, `overrun_excess` and the
   spend checks of `verify`, C15 against `account_readings`, `paid_key_refusal` and `key_differs`, C17 against
   `streak_closes`, `only_pre_request_appends`, `listing_check` (`max_completion_tokens`), `KEY_PATTERN` and
   `cloud_analysis` (`deviations`, `account`); and Correction 3: C18 against `_http_failure`, `_error_from_json`
   (`own_message` without the provider's raw text), `classify` (`FILTER_TYPES`, `ACCOUNT_TYPES`, the `forbidden`
   verdict) and the checks of `verify` on journaled filter refusals and key-limit stops, C19 against `classify`
   (`KEY_LIMIT_SOURCE`), `other_budgets`, `key_short`, `paid_key_refusal`, the key-short close in `_run` and
   `close_incomplete`, C20 against `filter_final`, `filtered_by_key`, the end of `_run`, the content-filter check of
   `verify` and `cloud_analysis.deviations`, C21 against `overrun_excess` and the overrun count in `verify`, `_num`,
   `error_count` and the transient branch of `_run`, and `_http_failure` (usage). Correction 3 quotes the
   documentation it rests on: re-read `api-reference/errors` and `api-reference/limits` on the day and compare.
   Correction 4: C22 against `filter_state`, `filter_waits`, the filter branch and the end of `_run` and the filter
   checks of `verify`; C23 against `forbidden_series`, `close_incomplete` (`--checked`) and the closing check of
   `verify`; C24 against `preflight` (the free queue before the paid arm); C25 against `key_use`, `key_use_stop`, the
   paid branch of `_run`, `_num`, `PAID_KEY_FIELDS`, `paid_key_config_refusal`, `arm_stopped_by_key` and
   `cloud_analysis.deviations`. Correction 4 quotes the same pages, read again on 2026-11-12 16:39 UTC.
3. Nothing in the amendment or its Correction states a result: no answer of any model of the branch exists
   (`results/models/cloud/` holds only `registration.json`, `pre_run_checklist.md` and `snapshot/`).
4. The twelfth-change entry in `docs/deviations.md`, the records in `results/post_run_changes.json` and the README's
   deviations paragraph say the same thing; `results/metrics.json` differs from `60dfe9b` only in the post-run change
   records (compare leaf by leaf).

## B. The model selection
5. `PYTHONPATH=src .venv/bin/python -m judge.cloud_select --check` passes, and `--resolve` prints the registered list, the rejected free
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
    is journaled, and the branch cap keeps room for one overrun answer (C3). These tests pass:
    `test_the_model_sub_cap_stops_before_a_request_whose_worst_case_would_exceed_it`,
    `test_the_branch_cap_stops_and_no_paid_model_starts_after_it`, `test_the_branch_cap_keeps_room_for_one_overrun_answer`,
    `test_an_output_overrun_on_a_paid_model_is_journaled_closes_it_and_check_still_passes`,
    `test_a_cost_above_the_reservation_closes_a_paid_model`,
    `test_an_output_overrun_on_a_free_model_is_journaled_and_the_run_goes_on`,
    `test_make_check_accepts_a_model_over_its_cap_only_by_its_journaled_overrun` (C14: a model over its 5 USD cap
    passes only by its matching overrun),
    `test_a_quarantined_answer_above_its_reservation_is_an_overrun_and_closes_a_paid_model`,
    `test_an_error_inside_a_200_without_usage_is_charged_on_a_paid_model_only`,
    `test_an_overrun_line_written_twice_does_not_hide_overspend` (C21: one spend entry per overrun, one overrun per
    paid model) and `test_an_http_error_that_reports_usage_is_charged_by_it` (C21). A free model's reported cost is
    counted.
15. The key: read only in `read_key` when a request is sent; sent only as `Authorization` with
    `add_unredirected_header`; redirects not followed; errors scrubbed; `main` prints no traceback; nothing under
    `results/models/cloud/` matches `KEY_PATTERN`; the Makefile names neither the key nor its file. The reviewer does
    NOT open or print the real key file; at most `stat -f '%Sp' ~/.openrouter_key` (expect `-rw-------`).
16. The key readings (`account_readings`) are `GET /api/v1/key` only (no `/credits`, which needs a management key),
    call no model and cost nothing; the paid arm refuses a key whose `limit` is null or not exactly 15 USD, whose
    limit resets, whose reading leaves out `limit`, `limit_remaining`, `limit_reset` or `budgets` (or gives `budgets`
    as anything but a list), or whose `budgets` or `effective_budget` names a budget other than `key_limit` (C1, C15,
    C19, C25; `test_a_paid_model_refuses_a_reading_without_budgets`), and NaN, infinity or an integer too large for a
    float is no reading (`test_nan_and_infinity_are_not_key_readings`, C21;
    `test_huge_integers_are_not_key_readings`, C25); a key configured as registered whose `limit_remaining` is
    short, and whose use agrees with the branch's journal, closes the paid model by the budget, so the queue goes on;
    key use beyond the journal by more than 0.05 USD is a recorded deviation and closes nothing by itself (C25)
    (`test_a_short_remaining_key_limit_closes_the_paid_model_so_the_queue_goes_on`,
    `test_a_paid_model_refuses_a_key_under_a_guardrail_or_workspace_budget`, C19); a free run on a key whose reading
    differs from `key_reading` is a recorded deviation (C15). Confirm the field names against `api-reference/limits`
    on the day.

## E. Held-out, order, stops
17. No held-out complaint can be sent: `test_an_injected_held_out_id_sends_journals_and_spends_nothing` passes, and
    `X.guard` is called before every journal entry in `cloud._run`.
18. The runner refuses (read `preflight`). `test_preflight_refuses_each_registered_condition` covers an unclosed model
    of Amendment 1 or 2, a changed protected file, a dirty tree outside `results/models/cloud/`, uncommitted results of
    an earlier model, an unpublished HEAD, an earlier model of the queue not closed, a closed model, a paid model after
    the branch cap, an arm after its key limit was spent (C13; only with `limit_source`, C25) and unlocked labels, and
    `test_the_paid_arm_starts_only_after_every_free_model_is_closed` a paid model while a free one is open (C24).
    The other refusals are tested
    directly, not through `preflight`: an incomplete registration
    (`test_the_registration_refuses_while_empty_and_is_read_once_filled`), a missing or group-readable key file
    (`test_the_key_file_must_be_private_and_nothing_is_sent_without_it`), and a failed listing check, not the
    endpoint's status, which is recorded (C2; `test_the_listing_check_refuses_a_missing_model_a_changed_endpoint_or_a_higher_price`,
    `test_the_listing_check_compares_max_completion_tokens_on_paid_models`).
19. Free-tier handling: the daily allowance stops a run before OpenRouter's limit; a 429 with
    `X-RateLimit-Remaining: 0` on a free model is a pause, not an error, unless it is the per-minute limit (C10); a
    data-policy refusal is an account condition and cannot close a model. Errors coded 403 and 402 (C11–C13 as
    corrected by C18–C20), sorted by their structured fields only
    (`test_documented_403_and_402_shapes_are_sorted_by_their_structured_fields_only`,
    `test_403_and_402_are_sorted_by_what_they_report`): only `limit_source` `openrouter_key_limit`, on a 403 or a
    402, stops the arm, and it closes no model by itself
    (`test_a_spent_key_limit_stops_the_arm_and_never_closes_a_model_as_rejected`,
    `test_a_key_limit_spent_with_a_402_stops_the_arm_and_its_models_can_be_closed`); message text and the provider's
    raw text stop nothing (`test_a_content_filter_whose_raw_text_names_a_limit_stops_no_arm`); a 403 with
    `error_type` `content_policy_violation` or `refusal` fails that one request, the run goes on, and the request is
    sent to the filter exactly twice: once more by the first run at least 24 hours and 10 minutes after the first
    refusal (runs before then skip it), whose outcome is final, by the server's dates or, failing them, by the
    journal's local times, recorded (C22; `test_a_content_filter_403_fails_one_request_and_the_run_goes_on`,
    `test_content_filter_refusals_minutes_apart_do_not_close_a_model`,
    `test_a_filtered_request_is_sent_exactly_twice_whatever_the_number_of_runs`,
    `test_a_filter_refusal_without_a_server_date_becomes_final_by_the_local_clock`,
    `test_make_check_fails_on_a_send_before_the_filter_wait_and_on_a_third_refusal`); any other 403
    (`permission_denied`, a guardrail block, a CDN's page) stops the run as an infrastructure or permission condition
    and closes nothing by itself (`test_a_storm_of_other_403s_stops_each_run_and_never_closes_a_model`), until the same
    403 on the same request on 3 runs over at least 72 hours with no answer in between lets `--close-incomplete MODEL
    --checked "..."` close the model with a recorded deviation (C23;
    `test_a_persistent_forbidden_closes_the_model_only_after_3_runs_over_72_hours`,
    `test_a_persistent_forbidden_series_needs_the_span_and_no_answer_between`); a 400 `token_limit_exceeded` is an account
    condition (`test_a_token_limit_exceeded_400_is_an_account_condition_not_a_rejection`); the in-flight 402 is not one
    of the 150 errors (`test_the_in_flight_402_is_not_counted_among_the_error_answers`); only a moderation flag with
    `reasons` is scored as an answer.
20. Closing: only the conditions of §9 as corrected; the 24-hour rules read the server's `date` headers and need no
    answer (and, for the listing, no passing check) in between and hourly checks (C2); `make check` fails if the
    journals' times go backwards.

## F. The commands (check each command's own exit status, not a pipe's: lesson L-pipe of `docs/deviations.md`)
21. `make test` exits 0 (`make test; echo $?`).
22. `make check` exits 0.
23. `git status` is clean after the commit; no file outside `docs/`, `src/judge/`, `tests/`, `Makefile`,
    `results/models/cloud/`, `results/post_run_changes.json`, `results/metrics.json`, `results/report.md`, `README.md`
    changed.

## G. Before publication: the author's decisions (Correction 2, C16; filled in and committed before the amendment is published and the secret gist is created)
A "no" in item 26 or a "drop" in item 27 is made by a further correction of Amendment 3 before the gist, never after
it; until then the branch is registered as §2 lists it. The reviewer checks that every blank below is filled.
24. The key: a key created for this branch only, with a credit limit of **exactly 15 USD** and `limit_reset` null (it
    never resets), set in OpenRouter, preferably in a separate OpenRouter account or workspace (C1, C9, C15). The
    runner refuses a paid model on any other limit, on a resetting limit, on a reading that leaves out `limit`,
    `limit_remaining` or `limit_reset`, and on a key under a guardrail or workspace budget (C19).
    Key created as stated: ______ (yes / no); account or workspace: ______.
    The account's credit balance before the first request (C24: the documentation says a negative balance can refuse
    free models too): at least 17 USD, and at least 10 USD purchased in all (the 1,000 free requests a day):
    balance ______ USD, purchased in all ______ USD, read at ______ UTC.
    No guardrail (content filter, prompt-injection detector, allowlist or budget) assigned to the account (the
    workspace default), the member or the key (C25): ______ (yes / no).
25. The privacy setting for free and for paid models (C9), where it is set, and when:
    free: ______ paid: ______ set in: ______ at ______ UTC.
26. The paid arm (P1–P3; estimate 2.90 USD, worst case 10.13 USD, never more than the key's 15 USD):
    ______ (yes / no).
27. The reference row `google/gemma-4-31b-it:free` (600 free requests, about 0.6 of a day's quota; descriptive only,
    not compared): ______ (keep / drop).
28. One reading of `python -m judge.cloud --status` on the dedicated key (`limit` 15, `limit_reset` null,
    `budgets` with no scope other than `key_limit`, `free_model_daily_requests.limit`), copied into `registration.json` as `key_reading` and published with the
    registration (C10); the runner refuses while it is empty. Agreed: ______ (yes / no); reading taken at
    ______ UTC (filled at registration). If the free daily limit is 50, not 1,000, the schedule of §10 does not hold.

## H. The start day (the author and the operator, after publication and registration, before the first request)
29. The dedicated key of item 24 saved in the key file (`--key-file`, `OPENROUTER_KEY_FILE`, or the default in the
    author's home directory), readable only by its owner (`stat -f '%Sp'` gives `-rw-------`). A free run whose key
    reading differs from `key_reading` records a deviation (C15).
30. The privacy setting of item 25 is in effect on the account's settings page on the start day.
31. `python -m judge.cloud --check-listings` run on the start day, before the first request, to re-read the live
    listings of all seven models (C2); its records committed. A model whose check fails is not replaced; it waits, and
    `--watch-listing MODEL` decides whether it is ever closed.
32. The order of the arms is fixed (Correction 4, C24, replacing the operator's choice of Correction 3, C21): the free
    queue runs to its end before the first paid request, and the runner refuses a paid model while a free model is
    open. On the start day the balance of item 24 is still at least 17 USD and no guardrail has been assigned since.

## I. What the reviewer reports
A list of findings, each with severity (blocks publication / should fix / note), file and line, and evidence. Then a
single line: "publish" or "do not publish". The owner decides; nothing is pushed, no gist is created and no request is
sent before that.
