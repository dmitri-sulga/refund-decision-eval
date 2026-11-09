"""Single place for every setting that affects results.

Changing anything here changes the experiment; the report records these values.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE_FILE = RESULTS / "cache" / "responses.jsonl"
RUN_LOG = RESULTS / "run_log.jsonl"

# ---------------------------------------------------------------- data source
# The live CFPB API no longer serves consumer narratives (release 24,
# September 2026). Narratives come from a byte-identical mirror of the official
# bulk export taken on 2026-08-11; labels and metadata are cross-checked
# against the live official API (see data.py).
MIRROR_URL = ("https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints"
              "/resolve/main/complaints.csv.zip")
MIRROR_SHA256 = "d92239489715c816fa4fc77a27324878cfc88de5b417065ea185e80ac92a6658"
OFFICIAL_BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
LIVE_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

PRODUCT = "Credit card"
ISSUE = "Problem with a purchase shown on your statement"
DATE_MIN = "2023-01-01"  # inclusive (date received by the CFPB)
DATE_MAX = "2026-01-01"  # exclusive

# Label = the company's recorded response. 1 = monetary relief.
LABEL_MAP = {
    "Closed with monetary relief": 1,
    "Closed with explanation": 0,
    "Closed with non-monetary relief": 0,
}
# Every other status ("In progress", "Untimely response", legacy "Closed", ...)
# is dropped before sampling.

MAX_NARRATIVE_CHARS = 6000  # longer narratives are cut (same text for every model)

# ---------------------------------------------------------------- splits
SPLIT_SEED = 20261109
N_DEBUG = 12     # prompt smoke tests and timing only; never scored
N_DEV = 200      # iterate here
N_TEST = 250     # touched exactly once
N_TRAIN = 8000   # classic baseline only (the DEV learning curve is flat beyond ~2000)
NEAR_DUP_COSINE = 0.5   # word-trigram TF-IDF cosine; groups collapse to one record

# ---------------------------------------------------------------- model
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.8:27b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}
TOP_LOGPROBS = 5
REQUEST_TIMEOUT_S = 900

# ---------------------------------------------------------------- statistics
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
TARGET_AGREEMENT = 0.90   # pre-registered target for auto-decided cases
MIN_AUTOMATED = 20        # an operating point must automate at least this many DEV items

# ---------------------------------------------------------------- gaming test
N_GAMING = 30             # DEV items used for the adversarial rewrites
GAMING_SEED = 7
