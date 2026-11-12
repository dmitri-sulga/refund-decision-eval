# Cloud branch (Amendment 3): what the independent reviewer must verify before any request

Written with the amendment, before any request of the branch; updated with the amendment's Correction after the first
independent review (`a192234` reviewed; its findings B1, B2, M1–M7 and m1–m11 are answered by the Correction, C1–C10),
and with Correction 2 after the second (`c01188a` reviewed; its findings MAJOR-1 to MAJOR-3 and m-a to m-m are answered
by Correction 2, C11–C17), and with Correction 3 after the third (`0287331` reviewed; its findings MAJOR-1 to MAJOR-3
and m-1 to m-7 are answered by Correction 3, C18–C21, which also records the documentation it was checked against on
2026-11-12), and with Correction 4 after the fourth (`959a2db` reviewed; its findings MAJOR-1, MAJOR-2 and m-1 to m-7
are answered by Correction 4, C22–C25, with the documentation read again on 2026-11-12 16:39 UTC), and with Correction 5
after the fifth (`9001223` reviewed; its findings m-1 to m-6 are answered by Correction 5, C27) and after the author's
decision of 2026-11-12 to run without a credit limit on the key (C26, the owner's waiver), and with Correction 6 after
the sixth (`b05e4f9` reviewed; its finding MAJOR-1, auto top-up, and m-1 to m-7 are answered by Correction 6, C28,
which overrides C26's "the account's balance is the only outer bound"), and with Correction 7 after the seventh
(`65c3888` reviewed; its findings m-1 to m-5 are answered by Correction 7, C29, which records m-1 to m-3 and two limits
the review describes as known limits of the waiver, with the operator's checks in item 34, and changes no code).
The reviewer is a separate session that did not write this code. Every item is checked against the files and the public documentation, not against this list's wording. A
reviewer who finds a problem writes it down with the file and line; nothing is fixed silently, and any fix is made
before publication and recorded in `docs/deviations.md`.

## A. The plan and the record
1. `docs/analysis_plan_models.md`: lines 1–779 (lines 1–149, Amendment 1 and Amendment 2) are byte-identical to
   `60dfe9b`, lines 780–1094 (Amendment 3 as first reviewed) to `a192234`, lines 1095–1239 (the Correction as
   reviewed the second time) to `c01188a`, lines 1240–1344 (Correction 2 as reviewed the third time) to
   `0287331`, lines 1345–1474 (Correction 3 as reviewed the fourth time) to `959a2db`, lines 1475–1577 (Correction 4
   as reviewed the fifth time) to `9001223`, lines 1578–1669 (Correction 5 as reviewed the sixth time) to
   `b05e4f9`, and lines 1670–1753 (Correction 6 as reviewed the seventh time) to `65c3888`; each correction is only
   appended after them
   (`cmp <(git show 60dfe9b:docs/analysis_plan_models.md) <(head -n 779 docs/analysis_plan_models.md)`, the same with
   `a192234` and 1094, with `c01188a` and 1239, with `0287331` and 1344, with `959a2db` and 1474, with `9001223` and
   1577, with `b05e4f9` and 1669 and with `65c3888` and 1753, run in bash, not zsh, and
   `git diff 65c3888 -- docs/analysis_plan_models.md` has no removed line).
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
   Correction 2: C11 against `_error_from_json` and `classify` (`KEY_LIMIT_SOURCE`; `budget_spent` was removed by
   Correction 3), C12 against `filtered_counts`, the
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
   Correction 5: C26 against `key_limit_waiver`, `WAIVER_FIELDS`, `balance_refusal` (`BALANCE_MARGIN_USD`,
   `BALANCE_MAX_AGE_S`, `_utc_seconds`), `paid_key_config_refusal` and `paid_key_refusal` (`waived`), `key_use`
   (`usage` under the waiver), the paid branch of `_run` (the `key_limit_waived` deviation), the waiver checks of
   `verify`, `cloud_analysis.deviations`, `DEVIATION_WORDS`, `account` and `summary`, and `registration.json`
   (`key_limit_waived`, `key_limit_waiver`, `account_balance_readings`); C27 against `forbidden_series` (m-1), the
   closing check of `verify` (m-2), `cloud_analysis.deviations` and `deviations_sentence` (m-3), `DEVIATION_WORDS`
   (m-6), and items 2 and 27 of this list (m-5, m-4).
   Correction 6: C28 against `balance_refusal` (`auto_top_up` exactly false, no allowance ahead of the clock, the
   readings named by earlier waiver runs in `run_log`), `key_use` (`baseline`), the paid branch of `_run` (the usage
   baseline from `key_reading`, the key-use stop under the waiver, `key_differs` on a waiver run, the deviation's
   `auto_top_up` and baseline fields), `close_incomplete` (`under_waiver`), `key_limit_waiver` and `_waiver_date_ok`
   (`WAIVER_EARLIEST_DATE`, `WAIVER_DATE_AHEAD_S`), `paid_key_refusal` (no "None USD"), the waiver checks of `verify`
   (auto top-up off, one reading per paid run), `cloud_analysis.account`, `key_sentence` and `DEVIATION_WORDS`, the
   module docstrings, `registration.json` (`what`, `what_it_weakens`), and items 14, 24, 25, 26, 28, 30, 32 and 33 of
   this list.
   Correction 7: C29 changes no code; check that each known limit it records is what the code does (`key_use` and the
   paid branch of `_run` for m-1 and m-3, `key_differs` for m-2, the key-use stop under the waiver for the use of the
   key outside the branch) and that item 34 gives the operator's check for each.
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
    paid model), `test_an_http_error_that_reports_usage_is_charged_by_it` (C21) and, under the owner's waiver (no key
    limit), `test_under_the_waiver_the_in_code_caps_still_stop_a_probe_beyond_15_usd` (C26). A free model's reported
    cost is counted. Under the waiver this is the only bound of the paid arm in code (C26); outside the code only the
    account's balance bounds it, and only while auto top-up is off (Correction 6, C28): judge whether a defect could
    let the code's own accounting exceed 15 USD; a provider billing one answer beyond its reservation and the overrun
    room takes the branch over its cap by that answer (the test shows it), with no stop outside the code.
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
    on the day. Correction 5, C26: with the owner's waiver complete in `registration.json`, a key whose reading reports
    `limit` null may run the paid arm, only on a balance reading recorded there before the run, and each such run is a
    recorded deviation; a key that has a limit still needs exactly 15 USD that never resets, with or without the
    waiver; a key without a limit and without the waiver is still refused
    (`test_the_owners_waiver_lets_a_paid_model_run_on_a_key_without_a_limit_and_is_recorded`,
    `test_without_a_complete_waiver_a_key_without_a_limit_is_still_refused`,
    `test_the_waiver_never_admits_a_key_whose_limit_is_not_exactly_15_or_resets`,
    `test_under_the_waiver_each_paid_run_needs_a_fresh_balance_reading_that_covers_the_cap`,
    `test_make_check_fails_on_a_paid_run_without_the_limit_or_the_waivers_record`). Correction 6, C28: a balance
    reading counts only with `auto_top_up` exactly false
    (`test_under_the_waiver_a_balance_reading_must_record_auto_top_up_off`), never ahead of the clock and never for a
    second paid run (`test_a_balance_reading_ahead_of_the_clock_or_used_before_serves_no_paid_run`); under the waiver
    the key's use since `key_reading` (its `usage` the baseline) beyond the journal by more than 0.05 USD stops the
    paid run, and a missing baseline refuses it
    (`test_under_the_waiver_key_use_since_registration_beyond_the_journal_stops_the_paid_run`, which replaces C26's
    `test_under_the_waiver_key_use_is_read_from_usage_and_stops_nothing`); a paid run under the waiver on another key
    is a recorded deviation (`test_under_the_waiver_a_paid_run_on_another_key_is_a_recorded_deviation`); the waiver's
    date is a real date, not in the future (`test_the_waivers_date_must_be_a_real_date_not_in_the_future`); the key
    sentence never prints "None" (`test_the_key_sentence_never_prints_none_and_reports_each_guard`).

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
    `test_a_persistent_forbidden_series_needs_the_span_and_no_answer_between`; one undated refusal no longer moves a
    dated series to the local clock, C27: `test_one_undated_refusal_does_not_move_a_dated_forbidden_series_to_the_local_clock`;
    `make check` reads what the closing says it checked, C27:
    `test_make_check_reads_what_a_forbidden_closing_says_it_checked`); a 400 `token_limit_exceeded` is an account
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
Who fills each blank (Correction 6, C28, after the sixth review's section 7), marked after it:
- **[coordinator]**: copied by the coordinator from the read-only readings of the owner's OpenRouter dashboard taken on
  2026-11-12 at about 17:54 UTC (Credits, Guardrails and Privacy pages; nothing was changed), before publication. A
  time known only roughly is written as such ("about 17:54"), and a setting whose start is not known says so.
- **[owner]**: the owner's own decision; nothing in those readings answers it.
  The owner delegated these decisions to the coordinator on 2026-11-12 at 17:39:58 UTC, in a chat message to his AI
  coding assistant (dictated by voice; quoted as transcribed): «Короче, решаю все вопросы сам, с пользой Ропину
  Рутеров, что все, что тебе нужно, делай и решай вопрос, а работай» ("In short, decide every question yourself,
  [OpenRouter, garbled in the transcription]: whatever you need, do it, settle the question, and work"). The three
  [owner] blanks of items 24, 26 and 27 were filled by the coordinator under that delegation, marked
  [owner, delegated to the coordinator].
- **[registration]**: filled at registration from `python -m judge.cloud --status` (item 28), before any request.
- **[operator, start day]**: read again by the operator on the start day and before each paid run (section H).
24. The key and the guard of the paid arm (C1, C9, C15, C19; Correction 5, C26). One of:
    (a) a key created for this branch only, with a credit limit of **exactly 15 USD** and `limit_reset` null (it never
        resets), set in OpenRouter, preferably in a separate OpenRouter account or workspace; OpenRouter then refuses
        every request once 15 USD are used, a stop outside the code; or
    (b) the owner's waiver of a key limit (C26): the key has no limit (`limit` null), and `registration.json` records
        `key_limit_waived: true` with `key_limit_waiver` (his words in Russian and English, their meaning, the date,
        who gave them). Then **there is no stop outside the code**: the worst case is bounded by the code's caps only
        (15 USD for the branch, 5 USD per model, each request reserved at its worst case, the spend journal, the
        overrun room); a defect in the journal or the reservations could in principle exceed 15 USD, and a provider
        billing one answer beyond its reservation and the overrun room takes the branch over 15 USD by that answer.
        Outside the code only the account's credit balance bounds it, and only while the account's auto top-up is
        off (Correction 6, C28, overriding C26's "the only outer bound"): with auto top-up on, OpenRouter buys
        credits from the card when the balance falls below a threshold and nothing outside the code bounds the
        spend. Even with it off the bound is the balance as read before each paid run, not a fixed 25 USD (it can be
        topped up by hand), and OpenRouter may let it go below zero by about one request (the runner sends one
        request at a time).
        Auto top-up on the Credits page: **off** (read on 2026-11-12 at about 17:54 UTC with the 25.00 USD balance:
        the "Auto Top-Up" block offers "Enable"; nothing was changed) [coordinator]. It is read again with every
        balance reading of item 33, and the runner refuses a reading without `"auto_top_up": false`.
    The runner refuses a paid model on a limit other than exactly 15 USD (with or without the waiver), on a resetting
    limit, on a reading that leaves out `limit`, `limit_remaining`, `limit_reset` or `budgets`, on a key under a
    guardrail or workspace budget (C19, C25), and on a key without a limit unless the waiver is recorded completely
    (C26).
    Guard: (b), the owner's waiver, given on 2026-11-12 and recorded in `registration.json` and in C26 (his words
    there, verbatim). If a limit of exactly 15 USD that never resets is later set on the key, (a) applies by itself.
    Key created for this branch only: no (yes / no) [owner, delegated to the coordinator]; account or workspace:
    Personal (one account) [coordinator].
    What a "no" costs (Correction 7, C29, the seventh review's m-5): with no key of its own and no limit on it, the
    paid arm runs on the waiver path with C29's known limits (item 34), and any use of the key outside the branch after
    the item 28 reading, beyond 0.05 USD in all, stops every later paid run and so loses the whole paid arm.
    The account's credit balance before the first request, read on the OpenRouter dashboard (C24: the documentation
    says a negative balance can refuse free models too; C26: the code cannot read it, `GET /api/v1/credits` needs a
    management key): at least 17 USD, and at least 10 USD purchased in all (the 1,000 free requests a day):
    balance 25.00 USD, purchased in all 25.00 USD (one transaction, 2026-11-12), read at about 17:50 UTC on
    2026-11-12 [coordinator; read again on the start day, item 32].
    No guardrail (content filter, prompt-injection detector, allowlist or budget) assigned to the account (the
    workspace default), the member or the key (C25): yes (the Guardrails page: "No workspace guardrails yet"; no
    account, member or key guardrail seen) [coordinator; `budgets` of the item 28 reading confirms it].
25. The privacy setting for free and for paid models (C9), where it is set, and when:
    free: "Allow free endpoints that train on request data" ON, "Allow free endpoints that publish prompts" OFF;
    paid: "Allow paid endpoints that train on request data" OFF (Zero Data Retention off, the 1% data discount off);
    set in: the Personal account's privacy settings, https://openrouter.ai/settings/privacy; at: in effect when read at
    about 17:50 UTC on 2026-11-12, set earlier, at a time that is not known [coordinator; if only the time it was read
    is known, write "in effect at about ... UTC, set earlier"].
    Decided in advance (Correction 6, C28, the sixth review's m-7), so that nothing is decided during the run: if a
    free model fails with a data-policy error (OpenRouter finds no endpoint that matches the account's data policy),
    the operator may turn on "Allow free endpoints that publish prompts" for free endpoints only (the prompts are
    public CFPB complaint texts; the paid setting is not changed), and records a deviation first: a dated line under
    item 30 (the setting, the UTC time, the model and its error), committed before the next request of that model;
    the report's write-up lists it. `docs/deviations.md` is not edited during the run (its hash is recorded in
    `results/post_run_changes.json` and checked by `make check`, and `preflight` refuses uncommitted changes outside
    `results/models/cloud/`).
26. The paid arm (P1–P3; estimate 2.90 USD, worst case 10.13 USD, never more than the code's 15 USD cap by the code's
    own accounting; under the waiver of item 24 (b) there is no stop outside the code, see C26, and outside the code
    only the balance as read with auto top-up off, see C28): yes (yes / no) [owner, delegated to the coordinator].
27. The reference row `google/gemma-4-31b-it:free` (600 free requests, about 0.6 of a day's quota; descriptive only,
    not compared): keep (keep / drop) [owner, delegated to the coordinator].
    A "drop" must not block the paid arm (Correction 5, C27, the fifth review's m-4): the free queue must be closed
    before the first paid request (C24), and a row that is never run can neither run nor be closed. So "drop" is made
    only by a further Correction before the gist whose code also closes the row before any request ("not run: dropped
    by Correction N", in the run log, checked by `make check`) or takes it out of the C24 check of `preflight`, with
    a test that the paid arm then starts; until that Correction is committed and reviewed, the row is "keep".
28. One reading of `python -m judge.cloud --status` on the branch's key, copied into `registration.json` as
    `key_reading` and published with the registration (C10); the runner refuses while it is empty. It shows, under
    item 24 (a), `limit` 15 and `limit_reset` null; under (b), `limit` null and `limit_reset` null (C26); in both,
    `budgets` with no scope other than `key_limit` and `free_model_daily_requests.limit`; and, under (b), `usage` as a
    finite number, the baseline from which the key's use is compared with the spend journal (Correction 6, C28; the
    runner refuses a paid run under the waiver without it). Agreed: ______ (yes / no); guard shown by the reading:
    ______ ((a) / (b)); reading taken at ______ UTC [registration]. If the free daily limit is 50, not 1,000, the
    schedule of §10 does not hold. The dashboard readings of 2026-11-12 (limit none, `is_free_tier` false) do not
    replace this reading: they hold no `budgets`, free daily limit or `usage`.

## H. The start day (the author and the operator, after publication and registration, before the first request)
29. The key of item 24 saved in the key file (`--key-file`, `OPENROUTER_KEY_FILE`, or the default in the author's
    home directory), readable only by its owner (`stat -f '%Sp'` gives `-rw-------`). A free run whose key reading
    differs from `key_reading` records a deviation (C15).
30. The privacy setting of item 25 is in effect on the account's settings page on the start day: ______ (yes / no)
    [operator, start day]. A change made under item 25's rule is recorded here first (setting, UTC time, model, its
    error): ______.
    2026-11-12, about 17:50 UTC [coordinator, before publication; not the start-day reading above]: the privacy
    settings of item 25 read on https://openrouter.ai/settings/privacy, read-only: free "train on request data" ON,
    free "publish prompts" OFF, paid "train on request data" OFF; nothing was changed.
31. `python -m judge.cloud --check-listings` run on the start day, before the first request, to re-read the live
    listings of all seven models (C2); its records committed. A model whose check fails is not replaced; it waits, and
    `--watch-listing MODEL` decides whether it is ever closed.
32. The order of the arms is fixed (Correction 4, C24, replacing the operator's choice of Correction 3, C21): the free
    queue runs to its end before the first paid request, and the runner refuses a paid model while a free model is
    open. On the start day the balance of item 24 is still at least 17 USD, auto top-up is still off, and no guardrail
    has been assigned since: ______ (yes / no) [operator, start day].
33. Under the waiver (item 24 (b), C26), before each paid run (each `--run` of a paid model, a resumed one included):
    the operator reads the account's credit balance and the auto top-up setting on the OpenRouter Credits page and
    appends `{"balance_usd": ..., "read_at": "YYYY-MM-DDTHH:MM:SSZ", "read_by": "...", "source": "OpenRouter
    dashboard", "auto_top_up": false}` to `account_balance_readings` in `registration.json`, then commits it (before a
    model's first request it must also be published: `preflight` refuses an uncommitted change and an unpublished
    HEAD). `read_at` is this machine's UTC time when the balance was read (`date -u +%Y-%m-%dT%H:%M:%SZ`); a reading
    ahead of the runner's clock by any amount is refused (Correction 6, C28). `auto_top_up` must be the JSON value
    `false`: `true`, a missing field, `"false"` or `0` is refused (C28); if auto top-up is on, turn it off first. The
    runner refuses unless the latest reading is later than the branch's last paid send and than the reading of every
    earlier paid run under the waiver (one reading serves one paid run, C28), at most 24 hours old, not in the future,
    and at least the branch's remaining room under 15 USD plus 2 USD; it records each paid run under the waiver as a
    deviation. Under the waiver the key's use since `key_reading` beyond the spend journal by more than 0.05 USD stops
    the paid run with a recorded deviation (C28: a journal edited or lost cannot restart the spend); the model is then
    closed with `--close-incomplete` or a further Correction decides. The code checks the reading's form, time and
    size, not its truth: a wrong figure typed in is not caught. The operator also watches the spend journal
    (`results/models/cloud/spend.jsonl`) against the dashboard after each paid run and stops on any difference. First
    paid run: balance ______ USD at ______ UTC, auto top-up ______ (off) [operator, start day].
34. The known limits of the waiver (Correction 7, C29: the seventh review's m-1, m-2, m-3 and two more), before each
    paid run under the waiver, with item 33; the code does not do these checks:
    (i) `python -m judge.cloud --status` prints `usage` as a finite number (if not, the key-use check of C28 is off:
        the paid run is not started and a further Correction decides);
    (ii) that `usage` is not lower than `key_reading.usage` in `registration.json` (one key's use never goes down; a
        lower figure means another key, which `key_differs` cannot tell from the registered one when neither has a
        limit: the paid run is not started), and the key file holds only the registered key;
    (iii) the key is used for nothing else from the item 28 reading to the end of the paid arm (use outside the branch
        beyond 0.05 USD in all stops every later paid run and loses the whole paid arm);
    (iv) the branch runs from one clone on one machine, one `--run` at a time (two at the same moment are each held
        only by their own caps);
    (v) a `key_use_exceeds_journal` deviation recorded with a refusal for another reason compares the key's all-time
        `usage`, not its use since registration: it is a false record, the other cause is removed, and the report's
        write-up names it as such.

## I. What the reviewer reports
A list of findings, each with severity (blocks publication / should fix / note), file and line, and evidence. Then a
single line: "publish" or "do not publish". The owner decides; nothing is pushed, no gist is created and no request is
sent before that.
