# Addendum: the same task on other local models (planned, not run)

Registered 2026-11-11T11:50:01+00:00 (UTC). The commit that adds this file is the record. When it was made,
none of the models below had been sent a complaint or any other prompt. They had only been downloaded with
`ollama pull`, and the run code does not exist yet. The extension is exploratory: it was planned after the
held-out run, after the blind labels were locked and read, and after every DEV result had been seen, and it cannot
change a frozen claim. The AI coding assistant that wrote this repository chose the models and wrote this plan
(and will write the code), at the author's request. The author asked for the extension and decided to publish its
result, whatever it is.

## The question

Does either of two results of the study depend on which local model is used? The study ran one model,
`qwen3.8:27b` (digest `47951e4a6661...`, Ollama 0.40.15), so both could belong to that model rather than the task:

- **A.** Neither prompt beats the issuer lookup, which never reads the complaint. On DEV, the paired kappa
  difference, judge minus the lookup, is v1 -0.198 [-0.359, -0.036] and v2 -0.248 [-0.424, -0.061]. On held-out,
  v1 was below the lookup and v2 could not be told apart from it.
- **B.** Judge v1 pays far more often than the companies did: 137 of 200 on DEV, where they paid 60; 176 of 250 on
  held-out, where they paid 75.

## Models

From the official Ollama library (read 2026-11-11 with curl and the registry API). Rules: official namespace;
open weights with a licence file in the package; at least two families other than Qwen, plus a much smaller Qwen
model to separate size from family; the default tag, 20 GB or less; a spread of sizes; an instruction model that
takes the same request.

| Tag | Family (architecture) | Parameters | Weights as pulled (runner) | Download | Licence | Digest after pull | Why |
|---|---|---|---|---|---|---|---|
| `qwen3.5:9b` | Qwen 3.5 (`qwen3_5`, as the study model) | 9.7B | NVFP4, MXFP8 embeddings (MLX) | 7.6 GB | Apache 2.0 | `d63f830f28d515fefea546f213fca7a75e9dd5176ef51a6deabc952d1c99a911` | the same family and build recipe at a third of the size: separates size from family |
| `gemma4:12b` | Google Gemma 4 (`gemma4_unified`) | 12.4B | NVFP4, FP8 KV cache, draft model (MLX) | 7.7 GB | Apache 2.0 | `c7597fc90b86aff6965317f37dd5465402284e26382d9b4bf2dfe3ef1491f4b6` | a second family, medium size, same runner as the study |
| `phi4:14b` | Microsoft Phi-4 (`phi3`) | 14.7B | GGUF Q4_K_M (ggml) | 9.1 GB | MIT | `ac896e5b8b34a1f4efa7b14d7520725140d5512484457fab45d2a4ea14c69dba` | a third family, medium size; no thinking mode |
| `mistral-small3.2:24b` | Mistral Small 3.2 (`mistral3`) | 24.0B | GGUF Q4_K_M (ggml) | 15.2 GB | Apache 2.0 | `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b` | a fourth family, near the study model's size; no thinking mode |
| `qwen3.8:27b` (reference) | Qwen 3.8 (`qwen3_5`) | 27.8B | NVFP4 (MLX) | 18.2 GB | Apache 2.0 | `47951e4a666180bc...` | already scored; neither run nor pulled again |

**What the library page says, and what was pulled** (digests from Ollama's `/api/tags`, as `llm.environment()`
records them; 39.6 GB in all). The tags pages list every default tag as GGUF Q4_K_M. That is what came for
`phi4:14b` and `mistral-small3.2:24b`, whose IDs and sizes match the pages. For `qwen3.5:9b` and `gemma4:12b`,
Ollama 0.40.15 on this laptop (Apple M4 Max, 36 GB, macOS) resolved the same tags to safetensors builds quantised
to NVFP4 with NVIDIA ModelOpt for its MLX runner, with other IDs and sizes (the page: 6.6 GB GGUF for
`qwen3.5:9b`). The study's `qwen3.8:27b` is a build of that kind too (`quantization nvfp4`, MLX), though the page
lists it as GGUF. So runner and quantisation go with the model: two new models share the study's runner, two run
on ggml. The builds planned are the ones pulled, identified by their digests. `qwen3.8:27b` must not be pulled
again: the library's tag now points to another build, and a pull would replace the study's.

**Rejected.** `gemma4:31b` (20.4 GB); `gemma4:26b` (Mixture-of-Experts, same family as `gemma4:12b`);
`olmo-3.1:32b` (its default tag is the reasoning variant); `gpt-oss:20b` (its reasoning cannot be switched off);
`muse-glimmer:30b` (no licence file in the package); `granite4.2:8b` (no chat template or renderer in the package,
thinking on by default in its own template); `llama3.1:8b` (a community licence with a use policy, where Apache and
MIT models were available); `ministral-3` (same family as `mistral-small3.2`); `qwen3.5:4b` (`9b` gives the size
contrast); "decision" models such as `d3`, `clef`, `pplx-decider` (not chat models; the last is a Qwen3.8-27B
fine-tune); cloud-only models and anything over 20 GB; every namespace other than the official library.

## What is held fixed

- **Prompts.** v1 and v2 as the frozen `src/judge/prompts.py` builds them, sent through `llm.request_body(...)` with
  only `"model"` replaced; a test puts `qwen3.8:27b` back and checks the frozen template hash in `results/frozen.json`.
- **Options.** `{"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}`, `think: false`, the same
  JSON schema in `format`, logprobs requested, one request at a time. Each request carries its own system message,
  so a package's default one (`mistral-small3.2` ships one) is not used. Package parameters that the options do not
  set stay as shipped, as for `qwen3.8:27b`, and are recorded (for example `presence_penalty` 1.5 and
  `draft_num_predict` 3 for `qwen3.5`; a draft model for `gemma4`).
- **Parsing.** `judgments.parse`, unchanged. An answer that is not valid JSON, or that has no boolean decision, is
  counted, scored as no money back and never automated (plan item 10).
- **The one exception allowed in advance.** If the server refuses a request because of `think`, for a model
  without the thinking capability (`phi4`, `mistral-small3.2`), that field is left out for that model only, and
  this is reported. Any other refusal drops the model, and the drop is reported. No replacement is chosen once any
  answer of this extension has been seen.

## Data

- **Used:** DEV (200 complaints, unmodified text) and the labelled sample (100 TRAIN complaints, labels locked).
  **Not used:** any held-out complaint, the gaming rewrites, filtered texts, the guarded pipeline, replays.
- **How the code refuses held-out complaints.** The allowed IDs are built only from `splits["dev"]` and
  `data/label_sample.json`, and checked to be disjoint from `splits["test"]`, whose ID hash must equal the one in
  `results/frozen.json`. The command line offers only `dev` and `labelled`. Every ID is checked again just before
  its request is journaled, and a test checks that a held-out ID in the work list stops the run with no request
  sent.

## What is computed (per model x prompt x split)

1. **Kappa against the record**, with the study's bootstrap (`metrics.bootstrap`, 10,000 resamples, seed 0).
   Accuracy is secondary.
2. **On DEV, the paired kappa difference**, judge minus the issuer lookup (`metrics.paired_difference`, same
   settings). Judge minus TF-IDF is secondary.
3. **Pays N of n**, next to the record's N, with the paired bootstrap interval (10,000 resamples, seed 0, complaints
   resampled) of the payout share minus the record's.
4. **On the labelled sample:** against the person on his 56 decided labels, kappa and accuracy with the same
   bootstrap, and the registered sensitivity line ("cannot tell" as no money back); against the record on all 100,
   kappa and pays.
5. **Malformed answers** (count and share, including any `done_reason` other than `stop`), and **seconds per
   call** (median and mean wall time; generation and prompt tokens per second).
6. **Descriptive only:** kappa between each model's decisions and `qwen3.8:27b`'s on the same prompt and cases;
   and the same figures for `qwen3.8:27b`, from the existing caches with no new call.

## "Holds" and "differs", fixed now

- **A, per model and prompt, on DEV.** The 95% paired interval of kappa(judge) minus kappa(issuer lookup). Entirely
  above zero: **differs** (the judge beats the lookup). Otherwise: **holds**, printed as "below" when the interval
  is entirely below zero and as "not distinguishable" when it is not.
- **B, per model, on DEV.** The 95% paired interval of v1's payout share minus the record's (60 of 200). Entirely
  above zero: **holds**. Otherwise: **differs**. The labelled sample (the record paid 34 of 100) is reported for B
  without a verdict.
- **Format rule.** A cell with more than 20 malformed answers in 200 gets "not interpretable (format)" and counts
  for neither outcome.
- **Overall.** "Among the models tried, the result does not depend on the model" is printed only if A holds for
  both prompts and B holds for every model with interpretable cells. Otherwise the report says "it depends on the
  model" and names the model, the prompt and the direction.
- **Multiplicity.** There are 8 comparisons for A (4 models x 2 prompts) and 4 for B. Each verdict uses its 95%
  interval, as the study does. Bonferroni intervals are printed beside them (99.375% for A, 98.75% for B). A
  "differs" that the corrected interval does not support is printed with "not after correction for 8 (or 4)
  comparisons". No combined test is run.

## What will not be concluded

- **No ranking and no "best model".** Differences between the new models are not tested.
- **DEV favours the study model.** DEV was used for every choice in the original study: v2 was written from
  `qwen3.8:27b`'s DEV disagreements. So DEV is optimistic for that model, not for the others.
- **Family is not isolated.** Runner and quantisation go with the model (MLX NVFP4 against ggml Q4_K_M), so a
  difference between families is not a difference of family alone. `qwen3.5:9b` against `qwen3.8:27b` shares the
  runner and the recipe, but it is also a different generation.
- **The samples are small.** "Holds" means "not shown to differ". At n = 200 a paired kappa interval is about 0.3
  wide, and the 56 decided labels give wider intervals still.
- **Held-out is out of scope**: its verdicts stay those of `qwen3.8:27b`. Each figure here is one run of one build,
  under one Ollama version on one laptop, and is agreement with what the company recorded, not correctness.

## One run per cell

- **Order.** Each cell (model x prompt x split) is run once: `qwen3.5:9b`, `gemma4:12b`, `phi4:14b`,
  `mistral-small3.2:24b`; within each model, v1 then v2, DEV then the labelled sample. 600 calls per model, 2,400
  in all.
- **Checks before each run.** The code refuses to start unless this file is committed and unchanged, Ollama is
  0.40.15, and the model's digest is the one in the table.
- **The journal.** Every request is written to it before it is sent. A crash is resumed from it: only requests
  never journaled are sent. A journaled request without an answer is not resent unless `--resend-unanswered` is
  given, and the report then states it.
- **No second attempts.** A finished cell is never run again; no tag, option or prompt changes after the first
  call; nothing is re-drawn after a result has been seen.

## Storage and reporting

- **New files only.** A cache, `results/models/responses.jsonl` (keyed `model|version|item`), a journal,
  `results/models/sent.jsonl`, and a run log, `results/models/run_log.jsonl`. Nothing is written to
  `results/cache/`, `results/run_log.jsonl`, `results/human_labels/` or `results/frozen.json`, and `make check`
  must pass with every frozen number unchanged.
- **Record and README.** The run, its code and its report section are recorded as a post-run change
  (`results/post_run_changes.json`, `docs/deviations.md`). Every new figure is marked "added later; exploratory"; the
  first screen gets at most one sentence, saying that the extension ran on DEV and the labelled sample only.

## Amendment 1 (2026-11-11, before any request to any model of this plan)

**Where this note is.** The review asked for one line under the title. A line there would shift lines 1–149, so the
note is here instead. Lines 1–149 above are kept byte-identical to the registered commit `162909f`, and a test and
`make check` both check this. Where this amendment replaces a registered passage, the amendment governs.

**Why it was written.**
- A review by a separate AI-agent session (not a person) read the registered plan before any request. It found:
  - the verdict words could report an undetected difference as independence from the model;
  - the options were not the same in effect;
  - a model could still be replaced or dropped at the operator's discretion;
  - nothing required the plan to be public before the first call.
- The AI coding assistant that wrote the plan checked each point against the sources before adopting it, and
  says below where it disagrees.
- The author asked for a wider comparison, "the top models this computer can run". Three more models are therefore
  added below (§12).
- When this was written, no model of this plan had been sent any request.

**§0 Registration time.**
- The header's "Registered 2026-11-11T11:50:01+00:00" is when the text was finished on the laptop, by the
  laptop's clock.
- The checkable record is the commit `162909f` (commit time 12:25:08 UTC, set by the same clock) and its publication
  on GitHub. The repository's activity record shows `main` moving to `162909f` at 2026-11-11T12:32:36Z.
- No model of this plan had been sent any request before either time. The Ollama server log shows only
  `qwen3.8:27b` loaded that day. Its last generation request was at 10:10 UTC, from other work, and none of the
  models below was ever loaded.

**§1 Labels instead of "holds" and "differs".** This replaces the bullets "A", "B" and "Overall" under "Holds and
differs".
- **A, per model and prompt, on DEV.** The 95% paired interval of kappa(judge) minus kappa(issuer lookup), labelled:
  - "above the issuer lookup" if it lies entirely above zero;
  - "below the issuer lookup" if it lies entirely below zero;
  - "not distinguishable from the issuer lookup" otherwise.
- **B, per model, on DEV.** The 95% paired interval of judge v1's payout share minus the record's (60 of 200),
  labelled:
  - "pays more often than the companies did";
  - "pays less often than the companies did";
  - "not distinguishable from the companies' rate".

  Each B label is printed with "pays N of 200, the companies 60".
- **Words not used.** "Holds", "differs", "depends on the model" and "does not depend on the model" are not used
  for any result of this extension.
- **"More often" is not "far more often".** The registered B describes 137 of 200 against 60.

**§2 What the design can detect** (printed with the results).
- **Source.** These figures come from the reviewer's simulation: 500 simulated studies per point, each with a
  2,000-resample interval, drawn from the 200 DEV (record, lookup) pairs. The scripts and their output are
  committed in `results/models/review/` (`power.py`, `power.json`; the input arrays were produced by
  `recompute.py`). The only change is that the reviewer's absolute working directory in the two scripts is
  replaced by `.`.
- **A at 95%.** With 200 complaints, a model whose true kappa is above the lookup's by about +0.16 to +0.18 is
  labelled "above" about half the time. By about +0.20 to +0.25, it is labelled "above" about 80% of the time.
- **A after correction.** The simulation used the level for 8 comparisons, where 80% needs about +0.23 to +0.32.
  At the 14 comparisons of this amendment the requirement is higher still.
- **Reading the ranges.** Each range runs over the simulated judges: paying 30% or 46%, and copying the lookup on
  none or 30% of cases. The figures are read from `power.json`, where the reviewer's own summary gave narrower
  ranges.
- **For comparison.** The best DEV kappa in the study was TF-IDF's 0.383, against the lookup's 0.318 (+0.065, so
  "above" about 10 to 14% of the time). "Not distinguishable" therefore does not mean "no better than the lookup".
- **B.** A v1 that pays about 77 of 200 is labelled "pays more often" about half the time; about 86 of 200, about
  80% of the time.

**§3 The summary sentence** (replaces "Overall"). The report prints one sentence built from the labels, and no
other wording:

> "On DEV, judge v1 paid more often than the companies did for k of m models; across the m models and both
> prompts, j of 2m comparisons with the issuer lookup were below it, a were above it and the rest were not
> distinguishable"

- It is followed by the list of models, in the order of §9.
- **Corrected levels.** There are 7 models, so 14 A-comparisons and 7 B-comparisons. Every "above" and "below" (A),
  and every "pays more" and "pays less" (B), is printed with "also at the corrected level" or "not at the corrected
  level": 99.643% for A and 99.286% for B. The reference row `qwen3.8:27b` gets the same treatment; its two DEV A
  intervals include zero at 99.643%.
- **If any A label is "above",** the sentence adds: "This is one run of one build; it was not compared with
  qwen3.8:27b; the model may have seen these complaints and their outcomes in training (see §10)."
- **Always added after the sentence:** "The design compares each model with the issuer lookup and with the
  record, not models with one another; it does not show that the results are independent of the model."
- **Who counts in m.** m counts only models whose four cells are complete and interpretable and whose labels are
  not format-sensitive (§5, §7). Excluded models are named, with the reason.
- **Code.** `src/judge/models_analysis.py` computes §1–§3 and §7 and was written before any request. On the
  study's own answers it reproduces the figures quoted on lines 17–19.

**§4 Sampling penalties and effective parameters** (replaces "Package parameters that the options do not set stay
as shipped, as for qwen3.8:27b").
- **Checked in the Ollama 0.40.15 source.** In `api/types.go`, `DefaultOptions` sets `repeat_penalty` 1.0,
  `presence_penalty` 0 and `frequency_penalty` 0. In `server/routes.go`, `modelOptions` applies, in order: those
  defaults, a safetensors package's `generation_config.json`, the package's parameters, then the request.
- **Where the reviewer was wrong.** The reviewer's assumption of a repeat penalty of 1.1 over 64 tokens is not the
  default of this version.
- **What the packages set, read with `/api/show`.**
  - No `generation_config.json` of these packages sets a penalty.
  - The only package that departs from neutral is `qwen3.5:9b`, with `presence_penalty` 1.5. The study model's
    package sets 0.
  - `glm-4.7-flash` sets `min_p` 0.01, which keeps the most likely token and so cannot change a greedy choice.
- **What every request now carries.** Every request to the seven models carries, in addition to the registered
  options, `"presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1`: the values the study model ran
  with. Nothing else changes.
- **The test.** A test puts `qwen3.8:27b` back and removes these three keys, and must get the frozen template hash.
- **The environment dump.** The full `/api/show` output of every model (parameters, template, system message,
  capabilities, details, Modelfile) is saved in `results/models/environment.json`. It is committed and published
  before the first call, and the runner refuses if the live `/api/show` differs.
- **Speculative decoding** is left as shipped and reported per model:
  - `gemma4:12b` and `gemma4:31b` ship a draft model;
  - `qwen3.5:9b` sets `draft_num_predict` 3;
  - the server log shows `qwen3.8:27b` drafting.

**§5 Refusals, interruptions and replacement** (replaces "The one exception allowed in advance").
- **No replacement.** After this amendment, no local model is replaced or added at any time, whether or not any
  answer exists.
- **What a refusal is:** an HTTP 4xx answer to a model's first request, whose message names a request field.
  - If the field is `think`, `logprobs` or `top_logprobs`, it is left out for that model only, and the report says
    so. No label uses logprobs.
  - Any other named field drops the model, and the report gives the server's message.
- **Everything else is an interruption, at any point:** a timeout, a 5xx, a lost connection, a runner crash,
  sleep, or a 4xx that names no field or comes after the first request.
  - The run stops and is resumed.
  - A request journaled without an answer is sent again once, with `--resend-unanswered`, and every resend is
    listed in the report.
  - A model is never dropped for an interruption.
- **Incomplete cells.** A cell still missing answers after that one resend is closed (`--close-incomplete`) and
  labelled "incomplete (k of n answered)". It gets no A or B label, and its answered cases are reported.

**§6 Public before the first call.**
- **What is published, and how.** This amendment, the run and analysis code, its tests, `environment.json` and
  the review's power scripts are committed and published to the public GitHub repository before the first
  request. Publication goes through GitHub's git-data API, and every object is checked against the local hash.
- **A new gist.** The commit id and the SHA-256 of these files are written to a new secret gist created for this
  extension. The existing gist `dc5fd5e1...` is the held-out run's record and is not edited.
- **`results/models/registration.json`.** The time GitHub records for the publication of the code commit and the
  gist's creation time are saved in this file. It is added in a second commit, which is also published.
- **The runner refuses to start unless:**
  - that file exists;
  - the protected files (this plan, the extension's code and tests, the frozen prompt, request, parsing and
    metric code, the `Makefile`, `environment.json`) equal the published code commit;
  - the tree is clean outside `results/models/`;
  - HEAD is the commit GitHub shows as `main`;
  - Ollama is 0.40.15 and each digest is the table's.
- **After each model's 600 requests,** `results/models/` is committed and published before the next model starts.
  The runner refuses to start a new model while `results/models/` has uncommitted changes.

**§7 Malformed answers** (adds to "Parsing" and "Format rule").
- **The scoring stays as frozen.** Malformed answers are scored as no money back, as in the study.
- **Three readings.** For every DEV cell with at least one malformed answer, the A and B labels are also computed
  with malformed answers counted as money back, and with them left out. If the three readings do not give the same
  label, the cell is printed as "format-sensitive" beside its label and left out of m (§3).
- **Thresholds.** A DEV cell is "not interpretable (format)" above 20 malformed answers of 200, as registered. A
  labelled-sample cell is "not interpretable (format)" above 10 of 100.
- **Prompt length.** A response whose prompt token count exceeds 7,680 (8,192 minus 512) counts as malformed. So
  does any `done_reason` other than `stop`.

**§8 Ollama version and updates.** Each finding below was read from the files and from the v0.40.15 source; no
setting was changed and Ollama was not restarted.
- **The state now.**
  - The Ollama app 0.40.15 has run since 2026-11-07, with automatic updates on (`auto_update_enabled` = 1 in the
    app's settings database).
  - It checks for updates every hour. On 2026-11-09 at 23:54 UTC it downloaded 0.40.16 to
    `~/Library/Caches/ollama/updates/`.
- **When an update installs** (`app/cmd/app/app.go`, `app/updater/`):
  - A running app never installs an update by itself.
  - A downloaded update is installed when someone clicks "Restart to update" in the menu bar.
  - It is also installed at the app's next start (log-in, restart, opening the app) while automatic updates are on.
    A start triggered by the command line defers it.
  - With automatic updates off, the app neither downloads updates nor installs them at start.
- **What the operator must do.**
  - Before the first request: turn automatic updates off in the Ollama app's settings.
  - During the run: do not quit or restart the app, click "Restart to update", log out, restart or shut down.
  - Keep the laptop on mains power with the lid open. `make models` runs under `caffeinate -i`, which prevents
    idle sleep but not sleep from closing the lid.
  - Let no other process use Ollama.
- **What the code enforces.** Every response records the Ollama version and the digest. The runner refuses, and
  `make check` fails, on any response whose version is not 0.40.15 or whose digest is not the table's.
- **If the version changes anyway.** 0.40.15 is reinstalled before resuming. If that is not possible, the
  remaining cells are closed as "incomplete", and nothing is sent under another version.

**§9 Order and the first screen.**
- **One order.** Every table, figure and list of this extension gives the models in this order, which is also the
  run order, and in no other:
  1. `qwen3.8:27b` (reference, existing answers)
  2. `qwen3.5:9b`
  3. `gemma4:12b`
  4. `phi4:14b`
  5. `mistral-small3.2:24b`
  6. `granite4.1:30b`
  7. `glm-4.7-flash:latest`
  8. `gemma4:31b`
- **Within a model:** (v1, DEV), (v1, labelled), (v2, DEV), (v2, labelled). Nothing is sorted by a result.
- **The first screen.** The README's first screen gets this sentence and no figure from the extension: "Added
  later, exploratory: the same two prompts were run on other local models, on DEV and the labelled sample only;
  see [the section]."

**§10 What will not be concluded** (adds to the registered list).
- **The prompts.** v1 was written before DEV, so v1's DEV figures are tuned to no model. v2 was written from
  `qwen3.8:27b`'s DEV disagreements, so on the other models it is a prompt tuned for a different model. A lower or
  "below" v2 figure there cannot be attributed to the model.
- **The comparator.** The issuer lookup's threshold was chosen after DEV.
- **Training data.** The CFPB publishes each company's response next to the narrative, so any of these models,
  `qwen3.8:27b` included, may have seen these complaints and their outcomes in training. This is not measured.
- **Model size.**
  - Only models that fit one laptop are covered: at most 24 GB to download, within about 27.6 GiB of GPU memory.
  - The largest added model (31.7B) is about the study model's size. Nothing is said about larger or hosted models.
  - Models whose reasoning cannot be switched off, such as `gpt-oss:20b`, and default tags that are reasoning
    variants, such as `olmo-3.1:32b`, are excluded. The strongest reasoning models are therefore not covered.
- **Reproducibility.** Temperature 0 is greedy decoding, on which the seed has no effect. The MLX and ggml runners
  compute differently, and speculative decoding can change a token. The figures are therefore not reproducible by
  asking again; the study saw changed answers on DEV re-asks.
- **What the comparisons separate.**
  - Runner and quantisation go with the model.
  - `qwen3.5:9b` against `qwen3.8:27b` differs in size, generation and speculative decoding, so it does not
    separate size from family.
  - `gemma4:12b` against `gemma4:31b` (same family, generation, runner and quantisation type) is the closest
    contrast of size alone, but it is still one pair.
- **Timing.** Seconds per call are measured on a laptop used for other work, and are not compared between models.
- **The labelled sample.** All figures on it are descriptive, with no label. Where the two registered readings
  ("cannot tell" left out, or counted as no money back) disagree in sign or in the side of zero, the report says
  so and draws no conclusion.

**§11 Checks.** `make check` now also runs `python -m judge.models --verify`, which checks:
- that lines 1–149 of this file equal the registered commit;
- that `environment.json` holds Ollama 0.40.15 and the table's digests;
- that every cached answer has a journal entry for the same request, made before its `requested_at`;
- that no request was sent more often than §5 allows;
- that no journal entry or answer concerns a held-out complaint, or a model or prompt outside the plan;
- that every answer's model, digest, Ollama version and neutral penalties match its request and the table;
- that the first journal entry is later than the publication time in `registration.json`.

The analysis reports 4,200 answers, or the incomplete counts.

No request outside the 4,200 is sent to these models: no warm-up, no smoke test, no debug split. Before the first
request, the runner writes an extract of the Ollama server log (request paths and model loads only) to
`results/models/server_log_before_first_request.txt`, which is committed with the first model's results.

**§12 Three more models (the author's request).**
- **The rules for this round.** The registered rules, with two changes:
  - the download cap is raised to 24 GB, which fits the 27.6 GiB that Metal makes available to the GPU on this
    36 GB laptop;
  - at most 60 GB of new downloads.
- **What was preferred.** The largest suitable model of each family not yet in the set, and `gemma4:31b`, which
  now qualifies.
- **Request and pulls.** Every model gets the same request. Each was pulled with `ollama pull`, one at a time, on
  2026-11-11 between 13:21 and 13:53 UTC, 55.9 GB in all.
- **Expected memory** is the weights plus about 1 to 3 GiB at `num_ctx` 8192. This is estimated, not measured;
  the basis is `qwen3.8:27b`'s logged peak of 19.4 GiB for 16.9 GiB of weights.

| Tag | Family | Parameters | Weights as pulled (runner) | Download | Licence | Digest | Expected memory | Why |
|---|---|---|---|---|---|---|---|---|
| `granite4.1:30b` | IBM Granite 4.1 (`granite`) | 28.9B | GGUF Q4_K_M (ggml) | 17.5 GB | Apache 2.0 | `3f3e5df8a021439fd6f867a0e526bdc303cac79c811201cb6bac193298cb9fcd` | 17.5–19.5 GiB | IBM's largest dense model; template in the package; no thinking mode |
| `glm-4.7-flash:latest` | Z.ai GLM-4.7 (`glm4moelite`, mixture of experts) | 29.9B | GGUF Q4_K_M (ggml) | 19.0 GB | MIT | `4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6` | 19–21 GiB | the GLM family's largest that fits; thinking can be switched off (levels false, true) |
| `gemma4:31b` | Google Gemma 4 (`gemma4`) | 31.7B | NVFP4, draft model (MLX) | 19.4 GB | Apache 2.0 | `92808281a855666acf35743d49004bb30c810152cd004baf78f31b4578920e49` | 19.5–21.5 GiB | the largest Gemma 4 that fits; size contrast with `gemma4:12b` |

The earlier four, for the same estimate:

| Tag | Expected memory |
|---|---|
| `qwen3.5:9b` | 8–10 GiB |
| `gemma4:12b` | 8–10 GiB |
| `phi4:14b` | 9.5–11.5 GiB |
| `mistral-small3.2:24b` | 15–17 GiB |

**Rejected in this round.**
- `nemotron-3-nano:30b` (24.3 GB) and `nemotron-3.5-lightning:30b` (25.4 GB): over the cap.
- `qwen3.6:35b` (23 GB) and `qwen3.6:27b`: the Qwen family is already in the set.
- `olmo-3.1:32b`: the default tag is the reasoning variant.
- `command-r:35b` and `aya-expanse:32b` (CC BY-NC), and `exaone3.5:32b` (non-commercial EXAONE licence):
  non-commercial licences, where Apache and MIT models were available.
- `bonsai-2:27b`: a two-bit build of the Qwen architecture.
- `devstral-small-2`, `magistral`, `laguna-xs-2.1`, `north-mini-code-1.0` and `ornith-1.5:35b`: coding or
  reasoning models, or the same family as a model in the set.
- A fourth model: none fitted under the 60 GB total.

**Volumes and multiplicity.** Seven models make 4,200 requests (600 each). The multiplicity of §3 counts 14
comparisons for A and 7 for B.

**§13 A hosted arm.** A hosted arm (large open-weights models through a hosted service) may be registered later
as Amendment 2, before its own first call. Nothing in this amendment depends on it.

## Amendment 2 (2026-11-12, a hosted arm, registered before any request to it)

**Where this note is.** Below Amendment 1, which is unchanged, as are lines 1–149. This amendment adds a separately
registered arm. It replaces no passage of the plan or of Amendment 1, and nothing in them depends on it.

**Status.** No model below had been sent any request by this project's code, and no API key had been read by it, so
no hosted result could have shaped this amendment; the result is reported whatever it is. First drafted after the
first local model's 600 answers were committed, it was revised after four independent reviews by separate AI-agent
sessions while the local models were being run and published, and uses no result of Amendment 1. The author asked for
Anthropic's models to be compared; the AI coding assistant that wrote this amendment proposed the vendor's three
current tiers and the request settings below. The next paragraph is printed, word for word, with the results (§10).

**Same vendor.** The three models under test are Anthropic's. So is the AI coding assistant that wrote this
repository, drafted both judge prompts, proposed these three models and their settings, and wrote this amendment and
this arm's code; the model behind the earlier drafting of the prompts is not recorded here. The agent that wrote this
arm's code ran on `claude-opus-5-5`, one of the three models evaluated here, and so did every independent review of
this amendment; the coordinating assistant ran on `claude-fable-5-2`. The translation the annotator read beside each
labelled complaint was made by `claude-fable-5-2`, also Anthropic's, though not one of the three. This amendment is
published, and the hosted run starts, only after all seven local models are closed. This arm does not measure
whether any of this favours the vendor's models.

**§1 How this arm relates to Amendment 1.**
- **§13 of Amendment 1** foresaw "A hosted arm (large open-weights models through a hosted service)". This is a
  different one, three closed-weights models of one vendor through its API; no open-weights arm is registered.
- **§5 of Amendment 1** ("no local model is replaced or added at any time") concerns the local set. The hosted models
  are not added to it and do not enter m of its §3, its 14 A- or 7 B-comparisons, its summary or its levels.
- **§10 of Amendment 1** ("Nothing is said about larger or hosted models") stays true of every figure of Amendment 1;
  this arm has its own table, summary sentence and corrected levels (§10).
- **The exclusion rule.** The registered plan rejected `gpt-oss:20b` because "its reasoning cannot be switched off"
  (line 48). §10 of Amendment 1 reads: "Models whose reasoning cannot be switched off, such as `gpt-oss:20b`, and
  default tags that are reasoning variants, such as `olmo-3.1:32b`, are excluded." `claude-opus-5-5` is such a model:
  it rejects every setting that turns thinking off (§5). The rule is applied to the comparisons: `claude-opus-5-5` is
  not part of this arm's comparison set (§3), gets no label and is not counted in any sentence that counts models. It
  is still run, because the tier cannot be run any other way and the author asked for the vendor's models, and is
  shown as a separate, descriptive reference row labelled "reasoning on, cannot be disabled; 4096 max_tokens; not
  part of the comparisons" wherever it is printed.

**§2 Models.**
- **Rule.** One vendor's current small, medium and large tiers, chosen before any call. No model is added,
  replaced or dropped afterwards. They are run in this order, which is also the only order of every table of this
  arm, and each starts only if its projection fits what is left of the 15 USD (§7).

| Model | API ID sent | Tier | Input, USD per million tokens | Output, USD per million tokens | In the comparisons |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | `claude-haiku-5-5` | small | 0.10 (prompts up to 100,000 tokens) | 0.50 | yes |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | medium | 2.00 | 10.00 | yes |
| Claude Opus 5.5 | `claude-opus-5-5` | large | 4.00 | 20.00 | no: reference row (§1, §3) |

- **Where the IDs come from.** `https://platform.claude.com/docs/en/models/overview`, read on 2026-11-11 and
  2026-11-12. For these three models the API ID and its alias are the same string, with no date suffix; the page says
  that every Claude model ID, the dateless ones included, is a pinned snapshot.
- **Where the prices come from.** `https://platform.claude.com/docs/en/about-claude/pricing`: the standard prices on
  the Claude API. Cache writes are 0.125, 2.50 and 5.00 USD per million tokens, and cache reads 0.01, 0.10 and 0.20.
  No prompt of this study comes near 100,000 tokens, and the code refuses one that could.
- **Routing.** These prices are for global routing; a workspace may default to US-only inference, priced at 1.1 times
  (`https://platform.claude.com/docs/en/manage-claude/data-residency`). Every request therefore sends
  `inference_geo: "global"`, and the code records each answer's `usage.inference_geo` (the docs say that it reports
  where inference ran, but not what it reports for a request sent with `"global"`). Every answer is kept, whatever
  routing it reports; one that reports `"us"` is priced at 1.1 times, which the reserve of §7 covers. The number of
  answers that report a routing other than `"global"` is given per model and per cell (§10). Routing never closes a
  model.
- **Within a model:** (v1, DEV), (v1, labelled), (v2, DEV), (v2, labelled), as in Amendment 1, §9: 600 requests per
  model, 1,800 in all.

**§3 The comparison set and the reference row.**
- **The comparison set** is `claude-haiku-5-5` and `claude-sonnet-5-5`, the two models that run without reasoning,
  as the local models did. That gives **4 comparisons** with the issuer lookup (2 models x 2 prompts) and 2 payout
  comparisons. **Corrected levels**, fixed now for those counts: **98.75%** for A and **97.5%** for B.
- **Does Sonnet run without reasoning?** `claude-sonnet-5-5` does not accept `{"type": "disabled"}`. Its lowest
  setting, `{"type": "between_tools"}`, "turns off up-front thinking"; the model still writes progress updates
  between tool calls, and "Without tools, the response contains only text, as with `disabled` on Claude Sonnet 5"
  (`https://platform.claude.com/docs/en/build-with-claude/thinking`, "Turning thinking off"). This arm sends no
  tools, so by the documentation Sonnet runs without reasoning here. That is checked on every answer, not assumed: a
  model of the comparison set leaves the set and becomes a reference row, labelled "reasoning on (see Amendment 2,
  section 3); not part of the comparisons", if any of its answers reports `thinking_tokens` above 0
  (`usage.output_tokens_details.thinking_tokens`) or a `thinking` or `redacted_thinking` block. The rule is mechanical
  and fixed now; the docs say `thinking_tokens` may differ from the exact count by a few tokens, so even a count of 1
  moves the model. The corrected levels stay those fixed above.
- **The reference row,** `claude-opus-5-5`: its kappa against the record and against the person, its payout count,
  and the 95% paired intervals against the issuer lookup and the record are printed. It gets no A or B label, no
  corrected interval, and no place in the summary sentence's counts.

**§4 What is held fixed.**
- **Inputs.** The same complaints as Amendment 1: DEV (200, unmodified text) and the labelled sample (100), with the
  narrative every model sees (`prompts.visible_narrative`).
- **Prompts.** v1 and v2 as the frozen `src/judge/prompts.py` builds them: the system message as the request's
  `system` field and the user message as its one `messages` entry, byte for byte. A test puts the frozen Ollama
  request back together from what is sent and gets the frozen template hash in `results/frozen.json`. The model also
  sees format instructions that the API adds for structured output, which this project cannot read (§6).
- **Parsing, scoring and analysis,** unchanged: `judgments.parse` and `models_analysis.malformed` on the text blocks
  of the answer joined in order (an answer that is not valid JSON or has no boolean decision is counted, scored as no
  money back and never automated), and the functions of `src/judge/models_analysis.py`: the paired intervals, the
  three readings of malformed answers, the format thresholds (20 of 200, 10 of 100), the labels of Amendment 1 §1 and
  the person's figures. Only the comparison set and its corrected levels are this arm's own (§3).

**§5 The request.** Every request is `POST https://api.anthropic.com/v1/messages`, with the headers `x-api-key`,
`anthropic-version: 2026-10-21` and `content-type: application/json`, and this body:

| Field | `claude-haiku-5-5` | `claude-sonnet-5-5` | `claude-opus-5-5` |
|---|---|---|---|
| `model` | `claude-haiku-5-5` | `claude-sonnet-5-5` | `claude-opus-5-5` |
| `max_tokens` | 512 | 512 | 4096 |
| `system` | the frozen system message | the same | the same |
| `messages` | one user message, the frozen user text | the same | the same |
| `output_config.effort` | `"low"` | `"low"` | `"low"` |
| `output_config.format` | `{"type": "json_schema", "schema": S}` | the same | the same |
| `inference_geo` | `"global"` | `"global"` | `"global"` |
| `thinking` | `{"type": "disabled"}` | `{"type": "between_tools"}` | not sent |

S is the frozen JSON schema of the prompt version with `"additionalProperties": false` added to its object. Nothing
else is sent: no `temperature`, `top_p` or `top_k`, no `stop_sequences`, no tools, no `fallbacks`, no `speed`, no
`service_tier`, no `cache_control` and no streaming. No setting is changed after the first request (§9).

Why each (pages under `https://platform.claude.com/docs/en/`, read on 2026-11-11 and 2026-11-12):
- **No sampling parameters.** `api/versioning`: under API version `2026-10-21`, a request that sends `temperature`,
  `top_p` or `top_k` returns a 400 error on every model; `build-with-claude/thinking`: on these three models a
  non-default value does so under earlier versions too. Temperature 0, used for the local models, cannot be sent.
- **Thinking: the least each model accepts.** The local models were sent `think: false`. `build-with-claude/thinking`
  (the per-model table and "Turning thinking off"): Claude Haiku 5.5 accepts `{"type": "disabled"}` at effort `high`
  or below; Claude Sonnet 5.5 rejects `"disabled"` with a 400 error, and `{"type": "between_tools"}` is its lowest
  setting (§3); Claude Opus 5.5 rejects `"disabled"`, `"between_tools"` and a token budget at every effort level, so
  the field is left out and the model runs adaptive thinking (it decides whether and how much to think). Its thinking
  tokens (`usage.output_tokens_details.thinking_tokens`) are recorded per answer and printed per cell.
- **Effort `low` for all three.** `build-with-claude/effort`: effort controls how much a model thinks and spends;
  `low` is the lowest level (the defaults are `medium` for Haiku 5.5 and Opus 5.5, `high` for Sonnet 5.5). It is the
  closest to the local runs, which had no reasoning step, and one setting for the whole arm.
- **`max_tokens`.** The local models had `num_predict` 512 for the answer; with thinking off, 512 bounds the same
  thing (the study model's longest answer to these 600 requests was 71 tokens). With thinking on, thinking counts
  toward `max_tokens` ("a hard limit on total output (thinking plus response text)", `build-with-claude/effort`), so
  the limit is 4096. If `claude-opus-5-5`'s thinking uses most of it, the answer stops with `max_tokens`, is malformed
  and is scored as no money back; such answers are counted per cell.
- **The rest.** Structured output replaces Ollama's `format` (`build-with-claude/structured-outputs`: available on
  all three models; objects must set `additionalProperties` to `false`; a refusal or a `max_tokens` stop can return
  text that does not match the schema). `inference_geo` is sent because a request that omits it follows the
  workspace's default (§2). No `fallbacks`: a refusal is a result (§9), never answered by another model. No batch:
  requests go one at a time, as for the local models, each journaled before it is sent. API version `2026-10-21` is
  the latest in `api/versioning`. The transport is Python's `urllib`, as for Ollama: no new dependency, no retry
  inside a client, no redirect followed.

**§6 Differences that cannot be removed (limitations).**
- **Hosted, not local.** The models run under the provider's weights, serving stack and safety systems. Its model IDs
  are documented as pinned snapshots, which this project cannot verify, and the serving stack and safety systems may
  change between requests. Each answer records the returned `model` string, its `request-id`, the routing and the time.
- **No seed and no temperature** (§5): the answers are samples at the provider's default sampling, so asking again
  can give another answer. This is not measured: no answered request is sent twice.
- **Thinking differs:** `claude-opus-5-5` may think first; the other two do not (§3, §5).
- **The format mechanism differs:** the provider adds its own format instructions (counted in `input_tokens`) and
  needs `additionalProperties: false`; Ollama needed neither. Compiling a schema on its first use adds latency, so the
  first call per prompt version makes the seconds-per-call figures larger.
- **Tokens differ.** Counts come from the provider's tokenizer. The malformed rule that compares the prompt with
  7,680 tokens (Amendment 1, §7) is applied to the provider's count; no prompt of this study comes near it.
- **No logprobs:** the token-probability confidence is missing for every answer of this arm. No label uses it.
- **Refusals.** The provider's safety systems may decline a request (`stop_reason: "refusal"`). v1 asks for a
  one-sentence `reason` field, and a request read as asking for the model's reasoning can be declined in the category
  `reasoning_extraction`. Refusals are counted with their category and scored as malformed answers (§9).
- **Data retention.** The complaints are public CFPB narratives; sending them to the provider is covered by its
  retention terms, which this project does not control.

**§7 Budget.** The whole arm has a hard cap of 15 USD, enforced in code (`src/judge/hosted.py`).
- **What the cap bounds and guarantees.** The computed spend: the usage-based cost of every answer, plus the worst
  case of every request that may have been billed without a usable answer, plus the reservation of any request whose
  outcome is not yet known. No request is sent that could take that total over 15 USD, and every charge is at most the
  worst case reserved before its request unless the answer's billed input exceeds the bound, which closes the model
  (§9); so the computed spend can exceed 15 USD only by the excess of such an answer over its reservation. The amount
  billed can exceed it only if the provider bills more than the documented prices applied to the returned usage, or
  bills error answers. This project does not rely on a spend limit set in the provider's Console. If one is reached,
  the API's answer is an account condition (§9).
- **Error answers are not charged:** 3xx and 4xx answers, 429 (rate limit), 529 (overload) and every other 5xx. This
  is an assumption: the docs do not say that error answers are billed. If the invoice shows otherwise, that is
  reported.
- **Possibly billed losses are charged at their worst case:** an answer that arrived but cannot be read or has an
  unexpected shape (at the price of its `usage` if that can be read); an answer from another model (at the price of
  its `usage`, not cached); a timeout or a lost connection after the request was sent, and an interrupt while waiting.
- **Cost of an answer:** from the returned `usage` and the prices of §2, for input, output (thinking included), cache
  writes and cache reads (nothing is cached on purpose), at 1.1 times if `usage.inference_geo` is `"us"`.
- **Write-ahead reservation.** Before each request is sent, its worst case is written to the spend journal as a
  reservation. The answer's cost or the charge that settles it names the reservation, an error answer releases it,
  and an answer's spend entry is written before the answer is cached. After a hard crash, an unsettled reservation
  counts against the cap, the next run or close charges it before anything else, and the report's cost counts any
  that is still open. No send is charged twice.
- **Before each model: does it fit?** The code starts a model only if the computed spend so far plus the projection
  of its remaining requests is at most 15 USD. The projection takes input as the request's UTF-8 bytes over 3, plus
  500 tokens for the format instructions (an assumption: their size is neither documented nor measured), and output
  as 512 tokens per request, far above the 71 of the study model's longest answer. The projections are 0.27 USD for
  `claude-haiku-5-5`, 5.43 USD for `claude-sonnet-5-5` and 10.84 USD for `claude-opus-5-5`, 16.54 USD in all. So
  `claude-sonnet-5-5` starts only if `claude-haiku-5-5` cost at most 15 − 5.43 = 9.57 USD, and `claude-opus-5-5`
  only if the first two together cost at most 15 − 10.84 = 4.16 USD; the estimates at the end give 1.41 to 2.40 USD.
  A model that does not fit is not started, is reported as "not run: projection exceeded remaining budget", and no
  request of this arm is sent after it.
- **Before each request,** its worst case: input as the request's UTF-8 bytes plus 2,000 tokens (no token covers less
  than a byte; the 2,000 cover the format instructions), output as `max_tokens`, all at the 1.1 times rate. If the
  committed spend plus that worst case exceeds the cap, the request is not sent and the budget stop applies.
- **The cap is not raised.** After a budget stop, no request of this arm is sent, under this amendment or any later
  one. The model stopped is closed as "incomplete (k of 600 answered): stopped by the 15 USD cap", never dropped, and
  its missing requests are never sent. The run order is fixed (§2), so a budget stop removes the last requests in
  that order (v2, then the labelled sample, of the last model), not a random subset.
- **The spend journal**, `results/models/hosted/spend.jsonl`, holds the projection before each model, every
  reservation and what settled it, every answer's `request-id`, routing, tokens (thinking included) and cost with the
  running total, every charge with its reason, and each model's totals when it closes. `make check` recomputes every
  cost from the stored usage and fails if the computed spend plus the open reservations is over the cap.
- **No other request:** no warm-up, no token counting, no Models API call and no batch.

**§8 Held-out complaints are never sent.** The arm uses the local runner's item sets and guard unchanged (Amendment
1, "How the code refuses held-out complaints"): the allowed IDs come only from `splits["dev"]` and
`data/label_sample.json`; they are checked to be disjoint from `splits["test"]`, whose ID hash must equal the frozen
one; each ID is checked again just before its request is journaled, and the command line offers no split. A test puts
a held-out ID at the head of the work list and checks that the run stops with no request sent, nothing journaled and
nothing spent. `make check` fails on any journal entry or answer for a held-out complaint.

**§9 Answers, interruptions, retries, closing.** This replaces, for this arm, the rules of Amendment 1, §5. No model
is ever dropped, and no registered parameter is changed after the first request.
- **An answer** is an HTTP 200 response that can be read, has the expected shape, and was served by the requested
  model. It is cached and never resent, whatever routing it reports (§2). It is malformed (Amendment 1, §7) when its
  text is not valid JSON or has no boolean decision, or when its `stop_reason` is not `end_turn`: `refusal` (its
  `stop_details` category is recorded), `max_tokens`, `model_context_window_exceeded` or any other value. Malformed
  answers are scored as no money back, as frozen, with the three readings of Amendment 1, §7.
- **Rate limits, overloads and server errors** (429 other than the spend cap below, and every 5xx, 529 included) are
  waited out: up to three retries of the same request within a run, after 30, 60 and 120 seconds, or the
  `retry-after` the answer gives if longer (at most 300 seconds). They are journaled, not charged (§7), not counted
  as a send, and never use up the one resend. After the third retry the run stops; the next run sends the request
  again as a first send.
- **Account conditions** stop the run at once: 401, 402, a 429 whose `error.details.error_code` is
  `enforced_spend_limit_reached` (a monthly spend cap), and a 400 whose message says that the credit balance is too
  low or that the API usage limits were reached. They are reported as account conditions, not as a rejection of the
  request, are not counted in the error bound, and never justify a close: the run resumes once the account is fixed.
- **Any other 3xx or 4xx answer is a rejection:** every other 400, whatever its wording and whichever request it
  answers, and 403, 404 and 413; redirects are not followed. The run stops and nothing is dropped. The status, the
  error type and the message, with anything shaped like a key removed, are journaled; the request was not served and
  keeps its sends.
- **Lost answers:** a timeout or a lost connection after sending, an interrupt while waiting, and an HTTP 200 whose
  body cannot be read or has an unexpected shape. The request may have been served and billed, so it is charged (§7),
  and it is sent again once, with `--resend-unanswered`; every resend is listed in the report. A request is served at
  most twice: its first send and one resend.
- **Error bound.** At most 60 error answers per model, counting only the waited-out ones (429 other than the spend
  cap, 529 and every other 5xx). Rejections and account conditions never count toward it.
- **Closing a model is never the operator's choice.** It happens only on a condition the code checks in the journal:
  - in the run itself: all 600 answers are in ("complete"); an answer from another model (charged, not cached); an
    answer whose billed input is above the bound the cap relies on (cached); the budget stop (§7); or the end of the
    work list with every missing request's one resend served;
  - with `--close-incomplete`, which takes the run lock and charges any reservation a crash left open, only when (a) a
    budget stop of the model is journaled, (b) the error bound was reached, or (c) the model's last run stopped on a
    rejection and the same request received the same HTTP status and error type on runs at least 24 hours apart.
    Otherwise it refuses; an account condition never satisfies (c). The 24 hours are measured by the server's `date`
    header, which the journal records for every answer and error answer. `make check` fails if the times of the
    journal, the run log or the spend journal go backwards; deliberate falsification of these files is outside what
    the checks can exclude.
  A closed model is never run again. A model closed without an answer is reported as "not run: " and its reason, one
  with answers as "incomplete (k of 600 answered): " and its reason.
- **Incomplete cells.** A cell still missing answers after a close is labelled "incomplete (k of n answered)". It gets
  no A or B label, and its answered cases are reported.

**§10 What is computed and reported.**
- **Per model, prompt and split,** exactly what Amendment 1 computes, for the comparison set: kappa against the record
  with its interval; on DEV, the paired difference with the issuer lookup and the A label; for v1 on DEV, the payout
  difference and the B label; on the labelled sample, the person's figures; malformed answers and seconds per call;
  kappa with `qwen3.8:27b`'s answers, as a descriptive figure. For a reference row, the same without labels (§3).
  Added for this arm, per cell and printed in the table: the thinking tokens (median and maximum), the number of
  answers that reported a routing other than `"global"`, and the stop reasons other than `end_turn` with the refusal
  categories. Each model's status is printed in plain words: "complete", "incomplete (k of 600 answered): " or "not
  run: " with the reason (§7, §9).
- **Corrected levels.** 98.75% for A and 97.5% for B, printed as in Amendment 1, §3.
- **The summary sentence**, built by `src/judge/hosted_analysis.py`, counts the comparison set only. While any model
  is not closed it gives no count: its first part then reads "Hosted arm, on DEV: not finished, so no count is given;
  not closed:" with each such model and its status. Its parts, in this order:
  1. A sentence built with Amendment 1's template, starting "Hosted arm, on DEV:".
  2. If any A label is "above", this sentence: "This is one run of each model as the provider served it on the
     dates of the run; the models may have seen these complaints and their outcomes in training (see Amendment 2,
     section 14)."
  3. "The design compares each model with the issuer lookup and with the record, not models with one another; it does
     not show that the results are independent of the model."
  4. Any model of the set not counted, with its status.
  5. The reference rows with their label and status.
  6. Only if `claude-opus-5-5` has answers, this sentence: "claude-opus-5-5 ran with adaptive thinking, which it does
     not allow to be turned off, at effort low and max_tokens 4096. Its thinking tokens are given per cell in the
     table. It is a reference row, not part of the comparisons." It is followed by "[models] ran without up-front
     thinking and max_tokens 512, as the local models did", naming only the models of the comparison set that have
     answers and are still in it.
  7. This framing sentence: "The one person who labelled the sample agreed with the record at kappa 0.202 [-0.054,
     0.456] on his 56 decided labels, so neither the record nor this person is a gold standard. Each model's
     closeness to the record and to this person is given with its interval; the models are not ranked, and no model
     is called the best."
  8. The "Same vendor" paragraph above the §1 heading, word for word: the code reads it from this file.
  9. The cost, from the spend journal: per model (answers, charges without an answer, tokens, USD, with any open
     reservation at its worst case) and in total, against the cap; then every resend and every charge without an
     answer, per model, with its key and recorded cause.
- **Where it is printed.** The README shows hosted figures only in one section, headed "Hosted arm (Amendment 2;
  added later, exploratory)" and placed directly after the local-models section: the table that
  `hosted_analysis.table_markdown` prints, with each reference row labelled in its row, then the summary sentence,
  which carries the Opus sentence (when it applies), the "Same vendor" paragraph and the cost, each once. The first
  screen gets at most this one sentence, and no hosted figure: "Added later, exploratory: the same two prompts were
  also run on three hosted Anthropic models, on DEV and the labelled sample only; see [the section]." The local table
  never includes hosted rows.

**§11 What will not be concluded** (adds to Amendment 1, §10, which applies to this arm too).
- **No ranking.** The hosted models are not ranked or tested against one another, and neither are hosted and local
  models. Agreement with the record is agreement with what the companies recorded. The record and the one labeller
  agree only weakly with each other, so a higher kappa against either is not "better at the task".
- **Size is not isolated.** The two compared tiers differ in more than size (their training is not disclosed), and
  the largest tier is a reference row with reasoning on.
- **One run, one date.** Each figure is one run of each model as the provider served it on the dates of the run.
- **Not comparable with the local runs as an experiment.** Sampling, the format mechanism and the tokenizer differ
  (§6). The hosted table is printed in its own section (§10), not merged into the local one.

**§12 Public before the first call.** As Amendment 1, §6, for this arm.
- **What is published first.** This amendment, `src/judge/hosted.py`, `src/judge/hosted_analysis.py`,
  `tests/test_hosted.py` and the `Makefile` are committed and published to the public repository before the first
  request. The existing gists are not edited.
- **The registration.** `results/models/hosted/registration.json` records the arm's code commit and the time GitHub
  gives for its publication. It is committed now with those fields empty, and the runner refuses while either is
  empty. The tests pass in both states, so no protected file needs editing after publication.
- **The runner refuses to start unless** every local model of Amendment 1 is closed; the registration is complete;
  the protected files (`PROTECTED` in `src/judge/hosted.py`, which includes the frozen prompt, request, parsing,
  metric, data and label code) equal the published commit; the tree is clean outside `results/models/hosted/`; HEAD
  is the commit GitHub shows as `main`; the earlier hosted models are closed and no budget stop has occurred; the
  blind labels are locked; and `ANTHROPIC_API_KEY` is set. With the key unset, `make models-hosted MODEL=<id>` is the
  operator's dry run: every other check runs, and nothing is sent or recorded. `python -m judge.hosted --estimate`
  prints the projections of §7 without any request.
- **After each model's requests,** `results/models/hosted/` is committed and published before the next model starts.
- **The run lock.** A run and `--close-incomplete` take `results/models/hosted/run.lock`, created exclusively, before
  they read any state, and hold it to the end, so on one machine two of them can neither send at the same time nor
  work from stale state. After a crash the lock stays: `python -m judge.hosted --break-stale-lock` removes it only
  when the process that wrote it no longer exists, and logs the removal. If that process number now belongs to
  another program, the operator removes `run.lock` by hand after checking that no run is going on.
- **The key** is read from `ANTHROPIC_API_KEY` only when a request is sent, sent only in the `x-api-key` header (never
  forwarded on a redirect), and never written to any file, log line or error message: errors are rebuilt from the
  response body with anything shaped like a key removed, and no traceback is printed. The `Makefile` names neither the
  key nor a file holding it. `tests/test_hosted.py` checks this against a mock server that echoes a fake key, and
  `make check` fails if a file under `results/models/hosted/` holds anything shaped like a key.

**§13 Checks.** `make check` also runs `python -m judge.hosted --verify`, which reads files only. It fails if the
model list, its order or the comparison set differs from §2 and §3, if a hosted model is a local one, or if a file of
the arm holds anything shaped like a key; and, once there are requests, if the registration is incomplete, if the
first request was journaled before the code was published, or if the journal, the cache or the spend journal breaks
§2, §5, §7, §8 or §9 (each check is listed in `verify` in `src/judge/hosted.py` and tested in
`tests/test_hosted.py`).

**§14 Training data.** As Amendment 1, §10: these models may have seen the complaints and their outcomes in
training. This is not measured.

**Projected cost** (no request was sent to measure it).
- **The measured basis.** The 600 requests of each model hold 496,310 prompt tokens by the study model's tokenizer
  (`prompt_eval_count` in the study's own runs of `qwen3.8:27b`); its answers to them hold 34,618 output tokens.
- **The assumptions.** The provider's count is 1.0 to 1.35 times that. Its format instructions add 0 to 500 tokens
  per request; their size is not documented, so the true figure may be higher. The output is the same.
- **The estimates,** at the prices of §2: `claude-haiku-5-5` 0.07–0.11 USD; `claude-sonnet-5-5` 1.34–2.29 USD;
  `claude-opus-5-5` 2.68–4.57 USD before its thinking, and each 100 thinking tokens per answer add 1.20 USD.
- **The code's own projection** before each model (bytes over 3, plus 500, with 512 output tokens): 0.27, 5.43 and
  10.84 USD. These sum to 16.54 USD, above the cap, so `claude-opus-5-5` starts only if the first two models together
  cost at most 4.16 USD (§7); the estimates above give 1.41 to 2.40 USD.
- **When the cap would stop Opus.** If `claude-opus-5-5` thinks for more than about 670 to 910 tokens per answer on
  average, the cap stops it before its 600th request, depending on where in those ranges the other figures fall. Its
  remaining cells are then reported as incomplete. Lost answers (timeouts, lost connections) are charged at their
  worst case, about 0.12 USD each on `claude-opus-5-5`, and so lower that figure.

## Amendment 3 (2026-11-12, a cloud branch through OpenRouter, registered before any request to it)

**Where this note is.** Below Amendment 2, which is unchanged, as are Amendment 1 and lines 1–149. This amendment adds
a separately registered branch. It replaces no passage of the plan or of the earlier amendments, and nothing in them
depends on it.

**Status and who did what.** No model below had been sent any request by this project's code, and the OpenRouter key
had not been read by it. Unlike Amendments 1 and 2, this one was written after both earlier arms had run and their
results had been reported (eleventh change); the question below is asked because of what they left open, and its
answer is reported whatever it is. The author asked for a cloud branch through OpenRouter, with the free models as the
main arm and at most three paid models, a 15 USD ceiling, and only work that is useful. The AI coding assistant that
wrote this repository wrote the selection rules, applied them in code to a snapshot of public listings, and wrote this
amendment, the runner, the analysis and their tests.

**§1 The question, and why.**
- **The question.** Amendment 1 ran seven local models of 9.7B to 31.7B parameters and Amendment 2 three closed-weights
  hosted models. Amendment 1, §10 says: "Nothing is said about larger or hosted models". This branch asks, for
  open-weights models larger than every model run so far (124.5B to 1,650.5B parameters), the same two questions: on
  DEV, is judge v1 or v2 above, below or not distinguishable from the issuer lookup (A), and does judge v1 pay more or
  less often than the companies did (B)?
- **Why it is worth asking.** If no model of any size or family beats a lookup that never reads the complaint, that is
  a statement about the task and the record more than about one model; if one does, that is the first such result of
  the study. The design detects the same effect sizes as Amendment 1 (its §2), so "not distinguishable" again does not
  mean "no better than the lookup".
- **Relation to the earlier amendments.** Amendment 1, §13 foresaw "A hosted arm (large open-weights models through a
  hosted service)"; Amendment 2 registered a different one. This is that open-weights arm, registered as Amendment 3.
  Its models are not added to the local set of Amendment 1 (§5 there) or to the hosted arm of Amendment 2: they do not
  enter those arms' counts, comparisons, corrected levels or summary sentences. This branch has its own (§3, §11).

**§2 Models: the rules, the snapshot, the list.**
- **The snapshot.** On 2026-11-12 between 11:50:17 and 11:52:11 UTC, `python -m judge.cloud_select --fetch` saved,
  without a key: OpenRouter's model list (`GET /api/v1/models`, 468 models, 13 of them free), the provider endpoints of
  every model that passed the model-level rules (`GET /api/v1/models/{id}/endpoints`, 117 files), and for each listed
  open-weights model the parameter count Hugging Face reports for its stored weights (`safetensors.total` from
  `https://huggingface.co/api/models/{repo}`; four repositories answered 401 and so have no count). The raw responses
  are committed in `results/models/cloud/snapshot/` with `fetch_log.json`; their SHA-256 (over the sorted files) is in
  `results/models/cloud/registration.json`.
- **The rules,** in `src/judge/cloud_select.py`; `python -m judge.cloud_select --check`, part of `make check`, fails
  unless they select from the committed snapshot exactly the list below. Common to both arms, a model must:
  1. not be Anthropic's, OpenAI's or a router (`openrouter/free`);
  2. list a Hugging Face id (open weights) for which Hugging Face reports a parameter count;
  3. take text and return only text;
  4. not have reasoning that cannot be switched off (`reasoning.mandatory` in the listing), as the plan's exclusion
     rule requires (line 48; Amendment 1, §10);
  5. not be a coding, safety, translation, OCR, extraction or role-play model, a "thinking" or "reasoning" variant, or
     abliterated or uncensored (words in the id, listed in `EXCLUDED_ID_WORDS`);
  6. not be a derivative of another model: a listing that says "built on" or "based on" followed by a linked model, or
     "fine-tune of", "fine-tuned from" or "merge of", or a Hugging Face `base_model` of another organisation;
  7. have more than 32B parameters: larger than every model already run (`gemma4:31b`, 31.7B; `qwen3.8:27b`, 27.8B).
- **Free arm** (the main arm): every free model (`:free`, prompt and completion price 0) that passes rules 1–7, at
  most five, by parameters; its endpoint is the one listed with status 0.
- **Serving-contrast row:** a free model whose weights were already run in Amendment 1 (rule 7 aside) is run as a
  descriptive reference row. One exists: `google/gemma-4-31b-it:free`, the weights of `gemma4:31b`.
- **Paid arm:** priced models without a variant suffix that pass rules 1–7, whose weights are not in the free arm or
  already run, with at least one endpoint that has status 0, supports `structured_outputs`, `temperature` and
  `max_tokens`, states a precision of fp8 or wider (fp8, bf16, fp16, fp32: not fp4, int4 or "unknown"), and costs at
  most 1.00 USD per million input tokens and 4.00 USD per million output tokens with no per-request fee. Per developer,
  the largest such model (ties: the newest); then the three largest of these. Had the estimate of §7 been above the
  cap, models would have been dropped from the smallest up; it is not. The chosen endpoint is the one with the lowest
  estimated cost of 600 requests (§7), then the wider precision, then the tag.
- **One correction before registration.** The first version of rule 6 matched the words "built on" anywhere, and so
  excluded `nvidia/nemotron-3-ultra-550b-a55b:free`, whose listing says "Built on a hybrid Transformer-Mamba
  mixture-of-experts architecture". The rule was narrowed to a linked model name before anything was registered or
  sent; no answer of any model of this branch existed.

| Order | Model (OpenRouter id) | Arm | Parameters (Hugging Face) | Licence (Hugging Face) | Pinned endpoint (provider, precision) | USD per million input / output | `seed` sent | Format field |
|---|---|---|---|---|---|---|---|---|
| F1 | `nvidia/nemotron-3-ultra-550b-a55b:free` | free | 560.5B (55B active) | other (NVIDIA) | `nvidia` (Nvidia, not stated) | 0 / 0 | yes | none: the endpoint supports neither |
| F2 | `dots-studio/dots-3-note-preview:free` | free | 288.4B (16B active) | Apache 2.0 | `atlas-cloud/fp8` (AtlasCloud, fp8) | 0 / 0 | no: not supported | `json_schema`, strict |
| F3 | `nvidia/nemotron-3.5-super-vl:free` | free | 124.5B (12B active) | other (NVIDIA) | `nvidia` (Nvidia, not stated) | 0 / 0 | yes | `json_schema`, strict |
| F4 | `google/gemma-4-31b-it:free` | free, reference row | 31.3B | Apache 2.0 | `google-ai-studio` (Google AI Studio, not stated) | 0 / 0 | yes | `json_object` |
| P1 | `deepseek/deepseek-v4-pro-0813` | paid | 1,650.5B | MIT | `novita/fp8` (Novita, fp8) | 0.99 / 2.97 | yes | `json_schema`, strict |
| P2 | `moonshotai/kimi-k2.6` | paid | 1,026.9B | other (Moonshot) | `crusoe/bf16` (Crusoe, bf16) | 0.70 / 3.50 | yes | `json_schema`, strict |
| P3 | `xiaomi/mimo-v2.6-pro` | paid | 1,024.2B | MIT | `gmicloud/bf16` (GMICloud, bf16) | 0.32625 / 0.6525 | yes | `json_schema`, strict |

- **Parameters** are Hugging Face's count of the stored weights (active parameters from the OpenRouter listing, where
  it gives them). For weights stored packed (Kimi K2.6 stores most of its weights as int32-packed int4) the count is of
  stored elements; the 32B threshold is far below every model of the comparison set, so this changes no choice.
- **Rejected free models,** with the first rule each fails: `baidu/ernie-5.5-flash-preview:free` (no parameter count:
  Hugging Face answered 401; its reasoning is also mandatory); `thinkingmachines/inkling-2-small:free` (no open weights
  listed); `liquid/lfm-3-1.7b:free`, `liquid/lfm-2.5-2.6b:free` and `recursive/flint-v1.0:free` (reasoning mandatory);
  `cohere/north-mini-code:free` (a coding model); `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (a reasoning
  variant); `nvidia/nemotron-3.5-lightning:free` (31.6B) and `google/gemma-4-26b-a4b-it:free` (25.8B), not larger than
  the models already run; `google/lyria-3-pro-preview` and `google/lyria-3-clip-preview` (audio output, no open
  weights); `openrouter/free` (a router).
- **The next paid models by the rule,** not run: `tencent/hy4-preview`, `z-ai/glm-5.1`, `minimax/minimax-m3`.
- **Fixed now.** No model is added, replaced or dropped after this registration. If a model or its pinned endpoint
  disappears or changes (§9), it is closed as the rules say and reported; no substitute is chosen.

**§3 The comparison set and the reference rows.**
- **The comparison set** is F1–F3 and P1–P3: **12 comparisons** with the issuer lookup (6 models x 2 prompts) and **6
  payout comparisons**. **Corrected levels**, fixed now for those counts: **99.583%** for A (1 − 0.05/12) and
  **99.167%** for B (1 − 0.05/6), printed as in Amendment 1, §3.
- **Reasoning, checked on every answer.** Every request asks for reasoning to be switched off (§5). A model of the set
  leaves it and becomes a reference row, labelled "reported reasoning (see Amendment 3, section 3); not part of the
  comparisons", if any of its answers reports a reasoning token (`usage.completion_tokens_details.reasoning_tokens`
  above 0) or returns reasoning text (`message.reasoning` or `message.reasoning_details`). The rule is mechanical: one
  token is enough. The corrected levels stay those fixed above.
- **F4, the serving-contrast row,** gets no label and is not counted. Printed for it: the figures every model gets, the
  95% paired intervals against the issuer lookup and the record (descriptive), and, per cell, the kappa and the number
  of identical decisions between its answers and the answers of `gemma4:31b` in Amendment 1 to the same complaints.
  This is the one measure in the study of how much serving (provider, precision, runtime, format mechanism) changes the
  answers of the same weights; it is one contrast, not a test-retest.

**§4 What is held fixed.** As Amendment 2, §4: DEV (200 complaints, unmodified text) and the labelled sample (100),
with `prompts.visible_narrative`; v1 and v2 as the frozen `src/judge/prompts.py` builds them, the system and user
messages sent byte for byte as the request's `messages` (a test rebuilds the frozen Ollama request from them and gets
the frozen template hash); `judgments.parse`, `models_analysis.malformed` (a `finish_reason` other than `stop` is
malformed) and the functions of `src/judge/models_analysis.py`, unchanged. Within a model: (v1, DEV), (v1, labelled),
(v2, DEV), (v2, labelled), 600 requests; 4,200 in all.

**§5 The request.** Every request is `POST https://openrouter.ai/api/v1/chat/completions` with an `Authorization`
header carrying the key and `content-type: application/json`, and this body:

| Field | Value |
|---|---|
| `model` | the OpenRouter id of §2 |
| `messages` | the frozen system message, then the frozen user message |
| `temperature` | 0 (the local runs' value) |
| `seed` | 42 (the local runs' value), only where the pinned endpoint supports it (not F2) |
| `max_tokens` | 512 (the local runs' `num_predict`) |
| `reasoning` | `{"effort": "none"}` |
| `response_format` | `{"type": "json_schema", "json_schema": {"name": "judge_v1" or "judge_v2", "strict": true, "schema": S}}` where the endpoint supports structured outputs; `{"type": "json_object"}` where it supports only JSON mode (F4); not sent where it supports neither (F1) |
| `provider` | `{"order": [the pinned endpoint's tag], "allow_fallbacks": false, "require_parameters": true}` |
| `stream` | `false` |

S is the frozen schema with `"additionalProperties": false` added, as in Amendment 2. Nothing else is sent: no
`top_p`, `top_k`, penalties, tools, `transforms`, `models` fallback list or `usage` flag (usage, with its cost, is
always returned). No setting changes after a model's first request, with the one exception of §9.

Why each (pages under `https://openrouter.ai/docs/`, read on 2026-11-12):
- **Temperature 0 and seed 42** are the local runs' settings; the hosted arm could not send them, this one can. A
  hosted stack may still not be deterministic at temperature 0 (§6).
- **`reasoning: {"effort": "none"}`**: `use-cases/reasoning-tokens`: "`"effort": "none"` - Disables reasoning
  entirely"; models whose reasoning is mandatory reject it, and rule 4 excluded them.
- **Structured output** as the local runs' `format` and the hosted arm's `output_config.format`:
  `features/structured-outputs` (the request shape, `strict: true`, support "determined per endpoint", and the advice
  to set `require_parameters`). Where the pinned endpoint cannot constrain the output, the frozen prompt's own "Reply
  with JSON only" is all the model gets, and its malformed answers are counted as the frozen rule says (§6).
- **The provider is pinned** (`features/provider-routing`: `order` accepts a full endpoint tag such as
  `deepinfra/fp8`; `allow_fallbacks: false` keeps the request off any other provider; `require_parameters: true` skips
  an endpoint that would ignore a parameter instead of failing). Every answer must report the pinned provider (§9).
- **One request at a time**, each journaled before it is sent, as in the earlier arms. Transport: Python's `urllib`,
  no client library, no retry inside it, no redirect followed.

**§6 Differences that cannot be removed (limits).**
- **Hosted, not local.** The weights run on the pinned provider's stack, at its precision (fp8 or bf16 where stated;
  F1, F3 and F4 do not state it), behind OpenRouter. Each answer records the returned model name, the provider, the
  generation id, the server's date and the time.
- **No test-retest on the cloud.** No answered request is sent twice. Temperature 0 does not guarantee identical
  answers on a hosted stack (batching, kernels); the study already saw changed answers on DEV re-asks of a local model.
  F4 against `gemma4:31b` is the only serving contrast, and it mixes provider, precision and format mechanism.
- **Free-tier instability.** Free endpoints are rate-limited by OpenRouter (20 requests per minute; 1,000 a day for
  accounts that bought at least 10 credits; `api-reference/limits`, read 2026-11-12, which does not say when the daily
  count resets or whether failed requests count) and by the provider, and can be withdrawn without notice (F2 lists an
  expiry date of 2026-12-31). The schedule (§10) and the rules of §9 handle this; a model that disappears is closed
  and reported, never replaced.
- **Provider routing.** Each model is pinned to one endpoint with fallbacks off, so a provider outage stops the run
  (§9) instead of moving it to another provider, precision or price. The endpoint chosen for a paid model is the
  cheapest that qualifies, not the developer's own (DeepSeek's model has no endpoint run by DeepSeek in the snapshot).
- **The format mechanism differs between models:** strict JSON schema (F2, F3, P1–P3), JSON mode (F4), or none (F1).
  `judgments.parse` is unchanged and strict: an answer wrapped in a code fence is malformed and scored as no money
  back. F1 may therefore have more malformed answers for a reason that is not about deciding; the three readings of
  Amendment 1, §7 and the format thresholds (20 of 200, 10 of 100) apply as registered.
- **Tokens** are counted by each provider's tokenizer; the 7,680-token rule of Amendment 1, §7 is applied to them.
- **Data.** The complaints are public CFPB narratives. Free endpoints may log or train on prompts; the account's privacy
  setting must allow the pinned free endpoints, or they refuse (an account condition, §9).
- **Training data.** As Amendment 1, §10: these models may have seen the complaints and their outcomes in training.
  Not measured.

**§7 Budget.** A hard cap of **15 USD for the branch** and **5 USD for each model**, enforced in code
(`src/judge/cloud.py`). The author reported 25 USD of credit with 0 used on the account before this amendment.
- **Cost of an answer:** the larger of the cost the API reports (`usage.cost`) and the registered endpoint prices of §2
  applied to the reported tokens (`prompt_tokens`, `completion_tokens`; reasoning tokens are output tokens). Both are
  recorded. A free model's answer costs 0 unless the API reports otherwise, and then it is counted.
- **Before each request,** its worst case: input as the request's UTF-8 bytes plus 2,000 tokens, output as
  `max_tokens`, at the registered prices; 0 on a free model. If the committed spend (computed spend plus open
  reservations) plus that worst case would exceed either cap, the request is not sent and the model is closed as
  "incomplete (k of 600 answered): stopped by the cap"; after a stop at the branch cap no paid request is sent.
- **Before each paid model:** its projection (bytes over 3 plus 500 tokens of input, 512 tokens of output per request)
  must fit under both caps, and the account's remaining credit, read from `GET /api/v1/credits`, must be at least the
  worst case of the model's remaining requests plus 1 USD; otherwise nothing is sent.
- **Write-ahead reservation, losses, crashes:** as Amendment 2, §7. A lost answer is charged at its reported usage if
  that can be read, otherwise at its worst case.
- **Credits before and after.** Every run records `GET /api/v1/credits` (total credits, total usage, remaining) and
  `GET /api/v1/key` (the free-model daily count) before its first request and after its last. These two requests use
  the key, call no model and cost nothing; their numbers are printed with the results.
- **The spend journal,** `results/models/cloud/spend.jsonl`: every estimate, reservation, settlement, answer (model,
  key, generation id, time, tokens, reported and charged cost, running total) and charge, and each model's totals.
- **The estimate** (no request was sent to measure it). Basis: the 600 requests of `claude-haiku-5-5` in Amendment 2
  held 964,520 input tokens (with that API's format instructions; the local models' prompts held 476,098 to 501,654) and
  40,400 output tokens (local: 30,383 to 33,885); with a 30% margin, 1,253,876 input and 52,520 output tokens per model.

| Model | Input: 1,253,876 tokens x price | Output: 52,520 tokens x price | Estimate | Code's projection | Worst case (every request at its bound) |
|---|---|---|---|---|---|
| F1–F4 (free) | 0 | 0 | 0 | 0 | 0 |
| P1 `deepseek/deepseek-v4-pro-0813` | x 0.99 = 1.2413 | x 2.97 = 0.1560 | 1.3973 USD | 2.1143 USD | 4.8145 USD |
| P2 `moonshotai/kimi-k2.6` | x 0.70 = 0.8777 | x 3.50 = 0.1838 | 1.0615 USD | 1.9239 USD | 3.8309 USD |
| P3 `xiaomi/mimo-v2.6-pro` | x 0.32625 = 0.4091 | x 0.6525 = 0.0343 | 0.4433 USD | 0.5961 USD | 1.4852 USD |
| **Total** |  |  | **2.90 USD** | 4.63 USD | 10.13 USD |

Prices are in USD per million tokens; `python -m judge.cloud --estimate` prints the last three columns from the
inputs. Each model's worst case is under its 5 USD cap and the three together under 15 USD, so no model was dropped
for cost; the caps can bind only through lost answers charged at their worst case, or if a provider bills more than
the registered price, which the listing check of §9 refuses before each run.

**§8 Held-out complaints are never sent.** The local runner's item sets and guard, unchanged (Amendment 2, §8); a test
puts a held-out ID at the head of the work list and checks that nothing is sent, journaled or spent; `make check`
fails on any journal entry or answer for a held-out complaint.

**§9 Answers, interruptions, retries, closing.** This replaces, for this branch, Amendment 1, §5. No model is ever
dropped at the operator's choice; no registered setting changes after the first request, except as stated here.
- **An answer** is an HTTP 200 with `choices[0].message`, `usage.prompt_tokens` and `usage.completion_tokens`, from
  the registered model (its id, its id without `:free`, or its canonical slug, with or without a date suffix) at the
  pinned provider. It is cached and never resent. It is malformed when its text does not parse or its `finish_reason`
  is not `stop` (`length`, `content_filter`, `error` or any other).
- **A moderation flag** (an error whose metadata gives `reasons`) is an answer with no text: cached, malformed, scored
  as no money back, its reasons kept; the run goes on.
- **The free daily limit** (a 429 on a free model with `X-RateLimit-Remaining: 0`) stops the run without counting as an
  error; the model resumes after the reset. A free run also stops itself, before any 429, after the day's remaining
  count reported by `GET /api/v1/key` less 25 (or, if that count is not reported, at 950 free sends in the UTC day by
  the journal).
- **Rate limits, overloads, timeouts reported by the server and server errors** (408, 429 otherwise, 5xx), including
  an error object inside an HTTP 200 with such a code: waited out, up to four retries of the same request after 30, 60,
  120 and 240 seconds (or a longer `retry-after`, at most 300); not charged unless the error reports usage; never using
  up the one resend. After the fourth the run stops; the next run sends the request again. At most 150 such errors per
  model; then the model may be closed.
- **Account conditions** stop the run at once and never close a model: 401, 402, a 4xx whose message names the data
  policy, a privacy setting or a guardrail, and a missing or group-readable key file.
- **Any other 3xx or 4xx is a rejection:** the run stops, nothing is dropped. **The one exception fixed now:** a 400 to
  a model's very first request whose message names `reasoning` or `seed` leaves that field out for that model only,
  from then on; every answer records the fields left out, and the report states them.
- **Lost answers** (a timeout or lost connection after sending, an unreadable 200): charged (§7), sent again once with
  `--resend-unanswered`; every resend is reported.
- **Listing check before each run** (public requests, no key): the model must still be listed with the same Hugging
  Face id, and its pinned endpoint with status 0, the same provider and precision, and a price no higher than
  registered (zero for a free model). If not, nothing is sent and the event is recorded with the server's date.
- **Closing a model** happens only on a condition the code checks in the files: all 600 answers ("complete"); an answer
  from another model or provider (charged, not cached); billed input above the bound the cap relies on (paid models);
  a cap; the end of the work list with every missing request's one resend served; or, with `--close-incomplete`, the
  error bound, or the same rejection of the same request, or the listing check failing, observed at least 24 hours
  apart by the server's date headers. A closed model is never run again; it is reported as "not run: " or "incomplete
  (k of 600 answered): " with its reason.

**§10 Schedule.** Two queues, each in the order of §2: free (F1, F2, F3, F4) and paid (P1, P2, P3). A model starts only
when the models before it in its queue are closed; the queues may alternate (paid models run while the free queue
waits for its daily reset), but never at the same time (one run lock).
- **Pace:** one request every 4 seconds on free models (15 a minute, under the 20 allowed), every second on paid ones.
- **Free arm:** 2,400 requests; at 975 a day: day 1, F1 (600) and 375 of F2; day 2, the rest of F2, F3 (600) and 150
  of F4; day 3, the rest of F4. At least three days of quota; provider rate limits and withdrawals can lengthen it.
- **Paid arm:** 1,800 requests, about two to four hours in all, in one day.
- **Start:** not before this amendment, the code and the snapshot are published, the secret gist is created and
  `results/models/cloud/registration.json` is completed and published (§12). After each model, `results/models/cloud/`
  is committed and published before the next model of either queue starts.

**§11 What is computed and reported, whatever the result.**
- **Per model, prompt and split,** what Amendment 1 computes (kappa against the record with its interval; on DEV, the
  paired difference with the issuer lookup and the A label; for v1 on DEV, the payout difference and the B label; on
  the labelled sample, the person's figures, descriptive; malformed answers and seconds per call; kappa with
  `qwen3.8:27b`'s answers, descriptive), with the corrected levels of §3. Added for this branch, per cell: reasoning
  tokens (median and maximum), finish reasons other than `stop`, moderation flags, the providers that answered; for F4,
  the serving contrast of §3. Each model's status in plain words.
- **The summary sentence,** built by `src/judge/cloud_analysis.py` from Amendment 1's template for the comparison set
  only, starting "Cloud branch, on DEV:"; while any model is not closed it gives no count and names the models not
  closed. Then: a sentence on one run of each model as served on the dates of the run, if any A label is "above"; "The
  design compares each model with the issuer lookup and with the record, not models with one another; it does not show
  that the results are independent of the model."; any model of the set not counted, with its reason; the reference
  rows; the person's framing sentence of Amendment 2, §10 (one source in the code); the cost per model and in total
  against the cap; the credit remaining by the API before the first run and after the last.
- **What will not be concluded** (adds to Amendment 1, §10 and Amendment 2, §11): no ranking and no "best model";
  these models are not compared with the local or the Anthropic models; size and family are not isolated (provider,
  precision and format mechanism go with each model); "not distinguishable" is not "no better"; each figure is one run
  of each model as served on the dates of the run.
- **Where it is printed.** A README section headed "Cloud branch (Amendment 3; added later, exploratory)", after the
  hosted section, with its detail in `results/report.md`; on the first screen at most: "Added later, exploratory: the
  same two prompts were also run on six larger open-weights models through OpenRouter, on DEV and the labelled sample
  only; see [the section]." The presentation code is written after the run, as in the eleventh change, and prints the
  registered analysis's output.

**§12 Public before the first call.** As Amendment 1, §6 and Amendment 2, §12.
- **Published first:** this amendment, `src/judge/cloud.py`, `src/judge/cloud_select.py`,
  `src/judge/cloud_analysis.py`, `tests/test_cloud.py`, the `Makefile`, the snapshot and
  `results/models/cloud/pre_run_checklist.md`, committed and published to the public repository through GitHub's
  git-data API, each object checked against the local hash, with `main` moved by a non-forced update.
- **A new secret gist** holds the code commit id and the SHA-256 of the published files and of the snapshot; the
  earlier gists are not edited. `results/models/cloud/registration.json` is committed now with the commit, publication
  and gist fields empty; they are filled from GitHub's records in a second commit, also published. The runner refuses
  while any is empty.
- **The runner refuses to start unless** lines 1–149 of this file are the registered text; every model of Amendments 1
  and 2 is closed; the registration is complete; the protected files (`PROTECTED` in `src/judge/cloud.py`: this plan,
  the branch's code and tests, the earlier runners, the frozen prompt, request, parsing, metric, data and label code,
  and the `Makefile`) equal the published commit; the tree is clean outside `results/models/cloud/`; HEAD is the commit
  GitHub shows as `main`; the earlier models of the queue are closed; the blind labels are locked; the key file exists
  and only its owner can read it; and the listing check passes.
- **The key** is read from the file named by `OPENROUTER_KEY_FILE` (by default a file in the author's home directory)
  only when a request is sent, sent only in the `Authorization` header (never forwarded on a redirect), and never
  written to a file, log line or error message; errors are rebuilt from the response body with the key and anything
  shaped like a key removed, and no traceback is printed. The `Makefile` does not name the key or its file. `make
check` fails if a file under `results/models/cloud/` holds anything shaped like a key.
- **The run lock** and its recovery are those of Amendment 2, §12 (`python -m judge.cloud --break-stale-lock`).

**§13 Checks.** `make check` also runs `python -m judge.cloud --verify` (files only: the model list, queues and
endpoints; no key in the branch's files; the time order of its journals; and, once there are requests, the
registration complete and the first request after publication; no held-out complaint; sends, retries and resends
within the rules; every answer sent as registered, from the registered model and provider, within `max_tokens`,
journaled before its request, with a spend entry equal to its recomputed cost; the running totals; both caps) and
`python -m judge.cloud_select --check`. `tests/test_cloud.py` tests each rule offline, with a fake transport, a mock
server on 127.0.0.1 and a fake key file.

**§14 Stop rules, in one place.** A request is not sent when it could take the spend over a cap. A run stops at the
free daily limit, at an account condition, at a rejection, after the fourth retry of a waited-out error, at a lost
answer, or when the listing check fails; none of these drops a model or changes a setting. A model is closed only as
§9 lists. No model is added or replaced, no cap is raised, and no request of this branch is sent outside the 4,200 of
§4: no warm-up, no test call, no probe of a model.

## Amendment 3, Correction (2026-11-12, after an independent review, before publication and before any request)

**Where this note is and why.** An independent review by a separate AI-agent session read the unpublished commit that
added Amendment 3 (`a192234`) and asked for the changes below before publication. Amendment 3 above is left exactly as
that review read it; where this Correction says otherwise, this Correction holds. Lines 1–149, Amendment 1 and
Amendment 2 are unchanged. When this was written nothing of the branch had been published, no gist existed, the
OpenRouter key had not been read by this project's code and no request of the branch had been sent; no answer of any
of its models existed. Pages under `https://openrouter.ai/docs/` were read again on 2026-11-12.

**C1. The budget guard reads the key's own limit, not the account's credit (replaces the credit readings of §7).**
- `GET /api/v1/credits` needs a management key (`api-reference/get-credits`: "Management key required", and a 403
  "Only management keys can perform this operation"); with an ordinary key the paid arm could not have started. It is
  not used.
- The branch uses a key of its own, created for it, with a credit limit set in OpenRouter by the author (at most
  15 USD, never resetting). It is OpenRouter's server-side ceiling on this branch's spend. The runner reads the key
  file named by `--key-file` or `OPENROUTER_KEY_FILE`, by default the file in the author's home directory (§12).
- Before each paid model, and before and after every run, `GET /api/v1/key` (`api-reference/limits`: `limit`, "Credit
  limit for the key, or null if unlimited"; `limit_remaining`; `limit_reset`, null when it never resets; `usage`, all
  time; `usage_daily`, the current UTC day; `is_free_tier`; `free_model_daily_requests` with `used`, `limit` and
  `remaining` in the current UTC day). A paid model starts only if `limit` is set and at most 15 USD, `limit_reset` is
  null, and `limit_remaining` is at least the worst case of the model's remaining requests plus 1 USD; otherwise
  nothing is sent. A free model may run on a key with no limit. The readings are recorded (not the key's `label`, which
  shows part of the key) and printed with the results in place of the credit figures of §7 and §11.
- The caps in code (15 USD for the branch, 5 USD for each model, §7) are unchanged and are the second guard.

**C2. The endpoint's status is recorded, not required; a model is closed for unavailability only after a full day
of hourly failures with no success in between (replaces the status condition of §9's listing check and the
listing condition of its closing rule).**
- The pinned endpoint of P2 (`crusoe/bf16`) was listed with status 0 in the snapshot (11:50–11:52 UTC) and with
  status −2 at 12:41 UTC the same day (the review's live check). A status read at the minute of a check would have
  let the time of the attempt decide.
- The listing check before each run still requires the model listed with the same Hugging Face id and its pinned
  endpoint listed with the same provider (compared as in C4), the same precision and no higher price. The endpoint's
  status is no longer a condition: it is recorded with the server's date on every check, and every change from the
  status last recorded (0 at the snapshot) is recorded as a status flip with its time. A request to an endpoint that is
  down fails and is handled by the rules of §9 (waited out, or a rejection).
- A failed listing check stops the run; it closes the model only if every listing check of the model since its last
  passing check and since its last answer failed, over at least 24 hours by the server's dates, with no two
  consecutive checks more than two hours apart. `python -m judge.cloud --watch-listing MODEL` makes one check an hour.
  A rejection closes a model only if its two observations 24 hours apart have no answer of the model between them.
- On the start day, before the first request, `python -m judge.cloud --check-listings` re-reads the live listings of
  all seven models and records the result. The snapshot is not re-taken and the list is not changed; the selection's
  own use of status 0 at the minute of the snapshot is stated here, not changed.

**C3. Output billed beyond `max_tokens` (corrects §7's statement of how the caps can bind, and §9).**
- `max_tokens` 512 is sent on every request; the code refuses to reserve for any other value. A provider may still
  bill more (reasoning counted outside the limit, for example). An answer billed with more than 512 output tokens,
  more input than the bound of §7, or a cost above its reservation is an **overrun**, a deviation type of its own: it
  is journaled (`kind: overrun` in `sent.jsonl`, with the tokens, the cost and the reservation), the answer is cached
  and scored as it is, and the report lists it. On a paid model it closes the model ("incomplete (k of 600 answered):
  billed output above max_tokens"); on a free model the run goes on. `make check` accepts an answer above `max_tokens`,
  or a cost above its reservation, only with its overrun entry, and fails if a paid model was sent a request after one.
- Before every paid request the branch cap also keeps room for one answer billed up to the endpoint's listed
  `max_completion_tokens` (P1 393,216 tokens, 1.1663 USD beyond the reservation; P2 235,929, 0.8240 USD; P3 945,000,
  0.6163 USD), so one overrun cannot take the branch over 15 USD; the registered worst case plus all three is
  12.74 USD. A model's 5 USD cap can be exceeded by at most that one answer, after which the model is closed.
- §7's sentence "the caps can bind only through lost answers ... or if a provider bills more than the registered
  price" is therefore incomplete: an overrun is a third way.

**C4. The served provider (replaces the provider part of §9's "An answer" and of its closing rule).**
- The reported provider is compared after case folding and removing spaces and punctuation, with the snapshot's
  provider name and with the provider part of the pinned tag (`NVIDIA` and `Nvidia`, `atlas-cloud` and `AtlasCloud`
  are the same). An answer without a provider field is accepted, by this explicit rule: the request pins the endpoint
  with fallbacks off; such answers are counted per cell and reported.
- An answer from another model or provider is charged, kept in `results/models/cloud/quarantine.jsonl`, never scored,
  and the run stops; the request is sent again on the next run, which does not use its one resend. A second such
  answer of the same model closes it.

**C5. Reasoning written into the answer's text (adds to §3).** A `<think>`, `<thinking>` or `<reasoning>` block in
`message.content`, closed or not, is reported reasoning under §3, as a reasoning token is; it is recorded per answer.
The text is parsed unchanged, so such an answer is usually malformed.

**C6. The registered outputs are computed by the registered code (adds to §11).** For every complete cell,
`src/judge/cloud_analysis.py` also computes items 1–4 of "What is computed" (lines 81–95), with the functions that
`src/judge/extension_late.py` applies to the earlier arms: accuracy against the record; on DEV, judge minus TF-IDF;
the payout share minus the record's with its interval; on the labelled sample, accuracy against the person (decided
labels, and "cannot tell" counted as no money back). No label is attached to them. Item 5 (tokens per second) is not
computed: the plan does not fix how per-call rates are aggregated and the answers carry no server timings; seconds per
call are given.

**C7. Factual errors in Amendment 3.**
- §6, "DeepSeek's model has no endpoint run by DeepSeek in the snapshot", is wrong: the snapshot lists `deepseek`
  (DeepSeek, status 0, 0.66 / 1.98 USD per million). It was not chosen because it states no precision ("unknown") and
  does not list `structured_outputs`, both required of a paid endpoint by §2.
- §6, the free daily limit: `api-reference/limits` describes `free_model_daily_requests` as counted in the current UTC
  day, so the count resets with the UTC day. It does not say whether failed free requests count.
- §2, on parameters: the sentence on Kimi K2.6's "int32-packed int4" weights is withdrawn; it contradicted itself.
  The counts are as Hugging Face reports them (`safetensors.total`). For packed or 8-bit storage (Kimi K2.6: 1,014.7B
  of its 1,026.9B stored as I32; MiMo-V2.6-Pro: 1,000.2B as U8; DeepSeek-V4-Pro-0813: 1,623.5B as I8) that count is
  not necessarily the number of parameters. No choice depends on it: the next paid model by size has 780.0B and every
  model of the set is far above 32B.
- §1 and rule 7, "larger than every model run so far": the sizes of Amendment 2's Claude models are not published.
  Read "larger than every open-weights model run so far".

**C8. "The first such result" (replaces the last clause of §1's "Why it is worth asking").** An "above" label at the
95% level is reported as it is; it is called a result of this branch only if it holds at the corrected level of §3
(99.583%) as well, and the summary sentence gives that count. The three families of comparisons (Amendment 1's 14,
Amendment 2's 4 and this branch's 12) are corrected each on its own and are not pooled into one familywise error rate
for the study.

**C9. Privacy and data (adds to §6, "Data"; a decision for the author before the first request).** OpenRouter's
setting that allows routing to providers that may train on, or log, prompts is on the account's settings page, with
"separate settings for paid and free models" (`guides/privacy/logging`); the page describes no per-key setting. A free
endpoint may need it allowed. Allowing it for this branch would allow it for every other use of the same account.
Recommended: a separate OpenRouter account (or a workspace, if OpenRouter offers one with its own setting) and its own
limited key for this branch (C1); otherwise allow it only while free models run and switch it back. The author records
the setting and its time in `results/models/cloud/pre_run_checklist.md` before the first request, and the report names
each free provider's data policy as listed. The complaints of DEV and the labelled sample may enter the training data
of third-party models, which affects any future re-run of this study; no held-out complaint is ever sent (§8).

**C10. Smaller points.**
- **The key in error messages.** Errors are cleaned of the key held in memory by the request that read it, and of
  anything shaped like a key; the key file is never read when no request is sent.
- **Per-minute and daily 429.** A 429 with `X-RateLimit-Remaining: 0` is the free daily limit (§9) unless
  `X-RateLimit-Limit` is the per-minute allowance (20) or the message names a minute; then it is waited out.
- **"13 of them free" (§2).** 13 models carry `:free`; 16 are listed at price 0. The three without `:free`
  (`google/lyria-3-pro-preview`, `google/lyria-3-clip-preview`, `openrouter/free`) are not free models by the rule and
  are listed among the rejected for completeness; `python -m judge.cloud_select --resolve` prints them separately.
- **The check of the selection.** `python -m judge.cloud_select --check` (in `make check`) also compares, for each
  model, the endpoint tag, provider, precision, prices, seed support, format field and `max_completion_tokens` with what
  the rules select from the snapshot.
- **Sensitivity of the paid list (§2).** P1 depends on the 1.00 USD input ceiling: `novita/fp8` costs 0.99. With a
  ceiling from 0.9553 to just under 0.99 the DeepSeek place goes to `deepseek/deepseek-v4-pro` (1,598.8B, an earlier
  release of the same weights family) on `streamlake/fp8`; below 0.9553 no DeepSeek model qualifies and
  `tencent/hy4-preview` (780.0B) is third. With an output ceiling under 3.50, Kimi K2.6 moves to `siliconflow/fp8`
  (no `seed`); at 3.00 the Kimi place goes to `moonshotai/kimi-k2-0905`. Requiring `structured_outputs` excludes
  `meituan/longcat-2.0` (1,775.6B); requiring a stated precision excludes DeepSeek's own endpoint. Any parameter
  threshold from 31.7B to 124B gives the same list. The ceilings are round numbers chosen with the snapshot in view;
  the words of `EXCLUDED_ID_WORDS` that name products (`devstral`, `schematron`, `ui-tars`, `venice`) come from this
  snapshot. The list stays as registered.
- **Registration.** The runner also refuses while `code_published_source`, `gist_file_sha256` or `key_reading` (the
  output of `python -m judge.cloud --status` when the branch is registered: the key's limit and the free daily limit)
  is empty in `results/models/cloud/registration.json`.
- **Schedule (§10).** "At least three days" holds only if the account has bought at least 10 USD of credits (1,000
  free requests a day); with less it is 50 a day, of which the runner leaves 25 unused, and the free arm
  would take about 96 days. The daily limit used is the one `key_reading` gives.
- **Listing checks and the clean tree.** A run log changed only by listing checks does not need a commit and a
  publication before a model that has not started can be tried again.
- **A paid model that leaves the set for reasoning** still runs its 600 requests, for a complete descriptive row,
  within the caps.
- **Checks (§13).** `make check` also checks the overrun entries (C3) and compares providers as in C4;
  `tests/test_cloud.py` also tests the refusals of the runner's preflight, the key limit, the status flips, the
  24-hour streak, the overrun, the quarantine, reasoning in the text, the 429 kinds and that no key file is read without
  a request.

## Amendment 3, Correction 2 (2026-11-12, after a second independent review, before publication and before any request)

**Where this note is and why.** A second independent review by a separate AI-agent session read the unpublished commit
that added the Correction (`c01188a`) and asked for the changes below before publication: three serious problems
(errors coded 403, the 5 USD check after an overrun, the author's decisions before the gist) and thirteen smaller
ones. Amendment 3 and its Correction above are left exactly as that review read them (lines 1–1239 of this file equal
`c01188a`); where this Correction 2 says otherwise, it holds over both. When this was written nothing of the branch
had been published, no gist existed, the OpenRouter key had not been read by this project's code and no request of the
branch had been sent. What it says of OpenRouter's errors is the review's reading of the documentation on 2026-11-12.

**C11. Errors coded 403 and 402 (replaces §9's "Account conditions" and "Any other 3xx or 4xx is a rejection" where
they differ).**
- From that reading: a spent budget (the key's own limit, a guardrail budget or a workspace budget) is refused with a
  403 "Budget limit exceeded (…)" and `metadata.limit_source`; a provider's content filter is a 403 with
  `metadata.error_type` `content_policy_violation` or `refusal`, the provider's name and its raw message, and no
  `reasons`; only OpenRouter's own moderation gives `reasons`; a 402 with `limit_source`
  `openrouter_in_flight_budget` is temporary and comes with `Retry-After`.
- An error is sorted in this order: `reasons` in its metadata, a moderation flag (§9, unchanged: an answer, cached,
  malformed, scored no money back, counted among the model's answers); a 403 with a `limit_source`, or whose message
  names a spent budget or limit, a budget stop (C13); a 402 with `limit_source` `openrouter_in_flight_budget`, waited
  out like a server error; the free daily limit (§9, C10); the account conditions of §9 (401, any other 402, a 4xx
  naming the data policy, a privacy setting or a guardrail, the key file); any other 403, a content-filter failure of
  that one request (C12); 408, 429 and 5xx, waited out; any other 3xx or 4xx, a rejection. **No 403 is a rejection, and
  no 403 closes a model by itself.**
- The error type that §9's 24-hour rule compares (the same request, status and error type) is `metadata.error_type`,
  as documented; `error.type` only where that is missing.

**C12. A provider's content filter (adds to §9).**
- A 403 sorted as a content-filter failure is a failure of that one request and a deviation type of its own
  (`kind: filtered` in `sent.jsonl`, with the status, the error type and the message, and a `filtered` event in the run
  log). It is not an answer, not a rejection and not one of the 150 errors; its reservation is released unless the
  error reports usage. The run goes on with the next request.
- The request is sent once more by a later run of the model (`--run`), not by the same run. Refused on two runs, it is
  not sent again; answered on the second, its answer counts as any answer does.
- When the work list ends and every missing request has either had its one resend served or been refused on two runs,
  the model is closed as "incomplete (k of 600 answered): … request(s) refused by the provider's content filter on 2
  runs". Under §11 an incomplete model of the comparison set is not counted, and a reference row stays descriptive. One
  complaint that a filter refuses on both runs therefore keeps a compared model out of the count. The review proposed
  scoring such a refusal as no money back, like a moderation flag; this Correction does not, so that a provider's
  filter is not scored as the model's decision.
- `make check` fails if a request is sent again after its second refusal.

**C13. A spent key limit is a budget stop of the arm (replaces, for this case, the account and rejection rules of §9).**
- A 403 reporting a spent budget stops the run and the arm in which it happened (free or paid): it is recorded
  (`key_limit_stop` in the run log, with the arm), and the runner sends no request of that arm after it. It closes no
  model by itself and never counts toward the 24-hour rule.
- A model of that arm not yet closed may then be closed with `--close-incomplete`, as "stopped by the budget: the
  branch key's limit in OpenRouter was spent" ("not run: " before it if it has no answer), as the caps of §7 close a
  model. The same now holds for a paid model that had not started when the branch reached its 15 USD cap, which the
  runner already refused to start and could not close. Neither is a rejection.

**C14. A model over its cap by an overrun (corrects §13 and the check of C3).**
- C3 allows a model's 5 USD cap to be exceeded by one overrun answer, while `make check` failed every model over
  5 USD, so the published check could not have passed after such an answer. `make check` now accepts a model over its
  5 USD cap only if a journaled overrun accounts for the excess: the overrun entry matches a charge in the spend
  journal (the same request, the same cost) against a reservation of the amount it records, and the model's spend less
  the overrun's cost beyond its reservation is at most 5 USD. Otherwise it fails.
- A charge above its reservation without an answer (an unreadable answer, or an error that reports usage) and an
  answer kept in quarantine that cost more than its reservation are overruns too: C3 applies (journaled, marked
  `without_answer` or `quarantined`; a paid model is closed). `make check` fails on any answer or charge above its
  reservation without its overrun entry.

**C15. The key (replaces "at most 15 USD" in C1).**
- The branch key's limit is set in OpenRouter to exactly 15 USD, with `limit_reset` null. The runner refuses a paid
  model on a key whose limit is not exactly 15 USD: a lower limit could refuse P3 near the worst case.
- A reading of `GET /api/v1/key` that leaves out `limit`, `limit_remaining` or `limit_reset` is refused for a paid
  model: a missing field is not read as null ("unlimited" or "never resets"). Every reading records which fields were
  missing, and also `budgets` and `effective_budget` (the guardrail and workspace budgets) where given.
- Before each free run the reading is compared with the registered `key_reading` (`limit`, `limit_reset`,
  `is_free_tier` where registered, and the free daily limit). If they differ, the free model still runs, and the
  difference is recorded as a deviation (`deviation`, `key_differs`, in the run log) and reported. The key's label is
  still not kept, so two keys with the same readings cannot be told apart.

**C16. The author's decisions before publication (adds to §12 and C9).** Before this amendment is published and the
secret gist is created, the author decides, and records in `results/models/cloud/pre_run_checklist.md` (its section
"Before publication"): the dedicated key with a limit of exactly 15 USD and `limit_reset` null (C1, C15); the privacy
setting for free and for paid models, where it is set and when (C9); the paid arm, yes or no; the reference row
`google/gemma-4-31b-it:free`, keep or drop; and that one reading of `python -m judge.cloud --status` will be copied
into `results/models/cloud/registration.json` as `key_reading` (C10). A "no" to the paid arm or a "drop" of the
reference row is made by a further correction of this amendment before the gist, never after it; until then the branch
is registered as §2 lists it.

**C17. Smaller points.**
- **The listing streak.** A listing check without a server date breaks a failing streak as a gap does; the latest
  unbroken run of dated checks decides (C2), so one undated check no longer blocks closing for ever.
- **Records written before any request.** A run log grown only by listing checks, key readings, a run started and
  paused or refused before its first send and a recorded key deviation, and a spend journal grown only by the
  estimate, need no commit and publication before a model that has not started can be tried again (extends C10's
  "Listing checks and the clean tree").
- **An error inside an HTTP 200 that reports no usage** is charged at the request's worst case on a paid model, since
  the provider may have billed it; on a free model it is released, as §9 says.
- **The listing check before each paid run** also requires the pinned endpoint's `max_completion_tokens` to be listed
  and no larger than registered: the room C3 keeps rests on it.
- **The report lists** each model's overruns, answers kept in quarantine, resends, content-filter refusals (and
  requests refused on two runs), fields left out, answers without a provider field and free runs on another key, as
  C3, C4 and §9 promise; `src/judge/cloud_analysis.py` prints them after the comparison sentence.
- **The key readings printed with the results** are those around the paid runs, before the first and after the last; a
  free run on the same key is not "the first paid run".
- **A key's label** (its first characters, three dots, its last ones) is treated as a key by the scrubbing of errors
  and by `make check`.
- **Wording.** The comment on the size threshold in `src/judge/cloud_select.py` now says "open-weights model" (C7).
- **Tests.** `tests/test_cloud.py` also tests each point of C11–C17, including the review's probes: a paid model's key
  limit spent (stopped, not closed, never a rejection), Gemma's request refused by a provider's filter (the run goes
  on), and a model 0.8917 USD over a sub-cap of 0.5 USD by its overrun (accepted only with the overrun).

## Amendment 3, Correction 3 (2026-11-12, after a third independent review, before publication and before any request)

**Where this note is and why.** A third independent review by a separate AI-agent session read only the change that
added Correction 2 (`c01188a..0287331`, unpublished) and found three serious problems in its sorting of errors and
seven smaller ones. Amendment 3, its Correction and Correction 2 are left exactly as that review read them (lines
1–1344 of this file equal `0287331`); where this Correction 3 says otherwise, it holds over all three. When this was
written nothing of the branch had been published, no gist existed, the OpenRouter key had not been read by this
project's code and no request of the branch had been sent. The author's decisions of C16 are unchanged and still open.

**The documentation this Correction rests on.** Read again for this Correction on 2026-11-12 at 15:26 UTC, with public
requests and no key: `https://openrouter.ai/docs/api-reference/errors.md` (served from
`https://openrouter.ai/docs/api_reference/errors-and-debugging.md`, 33,943 bytes, SHA-256 `3b7c66409cb68f8d…`) and
`https://openrouter.ai/docs/api-reference/limits.md` (served from `…/docs/api_reference/limits.md`, 19,661 bytes,
SHA-256 `a6c7d364fa408153…`). The words that decide the rules below:
- Error codes: "**403**: Forbidden (insufficient permissions, guardrail block, or moderation flag)".
- Typed error codes: "`permission_denied` | 403 | The key is valid but lacks the required permission or the request
  was blocked by a guardrail"; "`content_policy_violation` | 403 | The input or output was flagged by a content filter
  that runs separately from the model (provider- or OpenRouter-level), such as Gemini's `SAFETY` block"; "`refusal` |
  403 | The provider reported a model refusal as an error rather than as output"; "`token_limit_exceeded` | 400 | A
  token budget enforced by OpenRouter (e.g. credit-based cap) was exceeded."
- Guardrail errors: a request blocked by a guardrail gets "a `403` with a message describing the block reason"; the
  example is "Request blocked: prompt injection patterns detected", with `metadata.patterns` and no `error_type`.
- Budget errors: "When a guardrail budget, a workspace budget, or the key's own limit is spent, the request is rejected
  with a `403` before it reaches a provider"; `limit_source` is "`openrouter_guardrail_budget`,
  `openrouter_workspace_budget`, or `openrouter_key_limit`"; of `remedy_hint`: "branch on `limit_source`, not on this
  text".
- Handling 402 errors: "`openrouter_key_limit` means the API key's credit limit is exhausted. `openrouter_credits`
  means your balance cannot cover the request, or the single request is too expensive for your in-flight budget."
- The in-flight budget applies to "accounts whose balance is below a threshold, and, while the mechanism is being
  rolled out, newer accounts without an established spending history"; "It does not apply to requests to free models".
- Budget limits: OpenRouter "checks them in this order on every paid request"; "`limit` and `limit_remaining` keep
  describing only the key's own limit."
The two pages disagree on the status of a spent key limit (a 403 under "Budget errors", a 402 under "Handling 402
errors"); the rules below accept either.

**C18. Errors coded 403 (replaces, in C11, "a 403 with a `limit_source`, or whose message names a spent budget or
limit, a budget stop" and "any other 403, a content-filter failure").**
- An error is sorted in this order: `reasons` in its metadata, a moderation flag (§9, unchanged); a 402 or 403 whose
  `metadata.limit_source` is `openrouter_key_limit`, a budget stop of the arm (C19); a 402 whose `limit_source` is
  `openrouter_in_flight_budget`, waited out (C11); a 402 or 403 with any other `limit_source` (a guardrail or workspace
  budget, the account's balance), an account condition; the free daily limit (§9, C10); a 403 whose
  `metadata.error_type` is `content_policy_violation` or `refusal`, a content-filter refusal of that one request (C12,
  C20); 401, any other 402, a 400 whose `error_type` is `token_limit_exceeded`, and a 4xx whose own message names the
  data policy, a privacy setting or a guardrail, account conditions (§9); any other 403 (`permission_denied`, a
  guardrail block, a body that is not the API's JSON such as a proxy's or a CDN's page), an infrastructure or
  permission condition; 408, 429 and 5xx, waited out; any other 3xx or 4xx, a rejection.
- An infrastructure or permission condition is handled as an account condition: it is journaled as such
  (`kind: forbidden` in `sent.jsonl`, with the status, the error type and the message), the run stops, the request
  keeps its sends and a later `--run` tries it again; it never closes a model, never counts toward the 24-hour rule
  and is not one of the 150 errors. No 403 is a rejection, and no 403 other than the two filter types is a
  content-filter refusal.
- Message text never sorts an error into a budget stop or a content-filter refusal. The provider's raw payload
  (`metadata.raw`) is kept in the record but no longer sorts an error: the account phrases of §9 and the per-minute
  429 (C10) are read in OpenRouter's own message only. Only the first-request rule of §9, which leaves out a field the
  error names, still reads the raw payload, since a provider names its unsupported parameter there.
- `make check` fails on a content-filter refusal journaled with a status other than 403 or an error type other than
  these two, and on a key-limit stop (in `sent.jsonl` or in the run log) without `limit_source`
  `openrouter_key_limit`.

**C19. Budget stops (replaces C13's first point where they differ; adds to C1 and C15).**
- Only the branch key's own limit makes a budget stop of an arm: a 403 or a 402 whose `limit_source` is
  `openrouter_key_limit` (the documentation names both statuses). That limit is the registered cap (exactly 15 USD,
  never reset: C15) and cannot be raised. C13 then applies unchanged: the run and the arm stop, no request of that arm
  is sent after it, no model is closed by it, and `--close-incomplete` may close a model of that arm as "stopped by
  the budget".
- A guardrail or workspace budget and the account's balance (`openrouter_credits`) are the account's configuration,
  which the author can change (remove the budget, add credits), and a guardrail budget can reset; they are account
  conditions, which stop the run and close no model. The review proposed counting them as budget stops too; a budget
  stop is final for the whole arm, and this Correction keeps it for the one limit that cannot change.
- The paid arm also refuses to start on a key reading whose `budgets` or `effective_budget` names a budget other than
  the key's own limit (scope `key_limit`). This refusal closes nothing.
- A paid model whose key reading is as registered but whose `limit_remaining` is less than the worst case of the
  model's remaining requests plus 1 USD (C1) is closed by the budget before any request of that run ("stopped by the
  budget: the key's remaining limit in OpenRouter (… USD) did not cover the worst case of the model's remaining
  requests (… USD) plus 1.00", with "not run: " before it if it has no answer, and a `budget_stop` with scope `key` in
  the run log), as the caps of §7 close a model before it starts. Before, the runner only refused, so that model could
  neither start nor be closed and the paid models after it waited for ever. The next model of the queue is then tried
  with its own worst case. Every other refusal of the key reading (no limit, a limit other than 15 USD, a resetting
  limit, a missing field, a failed reading, another budget) stays a refusal that closes nothing: the author can
  correct it.
- `--close-incomplete` closes a model as stopped by the budget only after a key-limit stop of its arm recorded from
  the structured field, or after the branch cap (C13); no model that never ran is closed on a text match.

**C20. A provider's content filter: the 24-hour rule (replaces C12's second and third points where they differ).**
- A request refused by a content filter is sent again by a later run of the model, once per run, until it is answered
  or its refusal is final. The refusal is final when a second refusal comes at least 24 hours after the first by the
  server's date headers, as §9's rule for a repeated rejection requires (Correction, B2), so that the moment of a run
  cannot pick the result; a refusal without a server date does not count toward it. Runs minutes or hours apart close
  nothing.
- The model is closed, as C12 says, when the work list ends and every missing request has had its one resend served
  or has a final refusal; the reason reads "… request(s) refused by the provider's content filter on 2 runs at least 24
  hours apart". On a free model each later run spends one request of the day's quota per refused request.
- `make check` fails if a request is sent after its refusal became final.
- The review also proposed stopping a run as an account condition after five content-filter refusals in a row,
  without counting them. This Correction does not: with the 24-hour rule, and with other 403s no longer read as a
  filter, a quick series cannot close a model; refusals of the same requests that persist for a day are the filter's
  behaviour, which C12 reports; and a series stopped and never counted could leave a model that can never be closed.

**C21. Smaller points.**
- **One overrun, one charge.** `make check` matches each journaled overrun with one spend entry only, so an overrun
  line written twice cannot account for its excess twice, and it fails on a paid model with more than one overrun
  entry (the first closes it).
- **Finite numbers.** A key reading's `limit` and `limit_remaining`, and a listing's `max_completion_tokens`, count only
  as finite numbers: NaN and infinity, which a JSON reader may accept, fail "exactly 15 USD" and the remaining-limit
  check.
- **`token_limit_exceeded`** (a 400) is an account condition (C18), not a rejection.
- **The in-flight 402** is waited out as C11 says, journaled with its `limit_source`, and not counted among the 150
  error answers of §9: it reports the account's settling spend, which the documentation applies to newer accounts with
  a small balance, as the dedicated account of C9 may be. Its waits still count toward the four retries of one
  request, after which the run stops and can be resumed.
- **Usage on an HTTP error.** The body of an HTTP error is read for `usage`, as the body of an error inside a 200
  already was; a failed request whose error reports usage is charged by it, as C12 says ("released unless the error
  reports usage"); otherwise its reservation is released.
- **One key for both arms.** The free and the paid arm use the one dedicated key (C15, C16). The documentation says
  the budgets are checked "on every paid request" and that a spent key limit is refused "before it reaches a
  provider"; it does not say whether a spent key limit also refuses free models. If it does, an overrun of the paid
  arm that spends the key's 15 USD while free models are open stops the free arm too (C13), and its open models can
  then only be closed as stopped by the budget. The room kept under the branch cap for one overrun (C3) means this
  needs OpenRouter's accounting to exceed the code's own by more than that room. The schedule of §10 is not changed;
  the checklist names the order of the arms as the operator's choice that removes the risk (the free queue closed
  before the first paid request).
- **Not changed.** C12's choice stands: a final content-filter refusal is a deviation of that request, not an answer,
  and the report counts such refusals per model (C17). The review's suggestion of a sensitivity row that scores them
  as no money back is not added.
- **Tests.** `tests/test_cloud.py` tests each point of C18–C21, with the review's probes R1–R4 as tests: the documented
  403 and 402 shapes sorted by their structured fields only (R1); a series of 403s that are not a filter's stopping
  each run and closing nothing, and filter refusals minutes apart closing nothing (R2); a filter refusal whose raw text
  names a limit stopping no arm (R3); a spent key limit reported as a 402 stopping the arm, and a short remaining limit
  closing the paid model so that the queue goes on (R4); and each smaller point.
