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
