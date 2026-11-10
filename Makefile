PY := .venv/bin/python
export PYTHONPATH := src
MIRROR := https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints/resolve/main/complaints.csv.zip

.PHONY: venv test report check reproduce run replay freeze heldout label label-terminal judge-labelled data

$(PY):
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

venv: $(PY)

test: venv
	$(PY) -m pytest -q

# regenerate README blocks, results/metrics.json and results/report.md from the caches
report: venv
	$(PY) -m judge.report

# fail if any committed number differs from what the code regenerates
check: venv
	$(PY) -m judge.report --check

reproduce: test check

# model calls that are missing from the cache (DEV side only)
run: venv
	$(PY) -m judge.run --version v1 --split dev
	$(PY) -m judge.run --version v2 --split dev
	$(PY) -m judge.run --version v1 --gaming
	$(PY) -m judge.run --version v2 --gaming

replay: venv
	$(PY) -m judge.replay --version v2
	$(PY) -m judge.replay --version v1 --limit 100

# record the method and the DEV operating points before the held-out run
freeze: venv
	$(PY) -m judge.freeze

# the single held-out run (scored pass, then the noise pass): refuses unless the frozen method is
# unchanged and committed, refuses to run twice
heldout: venv
	$(PY) -m judge.run --final
	$(PY) -m judge.report

# blind labelling page in the browser (local only); resumable
label: venv
	$(PY) -m judge.label --annotator owner

label-terminal: venv
	$(PY) -m judge.label --annotator owner --terminal

# judge v1 and v2 on the labelled sample; refuses until the blind labels are locked
judge-labelled: venv
	$(PY) -m judge.run --version v1 --labelled
	$(PY) -m judge.run --version v2 --labelled

data: venv
	mkdir -p .download && curl -L -o .download/complaints.csv.zip $(MIRROR)
	$(PY) -m judge.data --mirror .download/complaints.csv.zip
