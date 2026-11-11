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
