PY := .venv/bin/python
export PYTHONPATH := src
MIRROR := https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints/resolve/main/complaints.csv.zip

.PHONY: venv test report check reproduce run data

$(PY):
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

venv: $(PY)

test: venv
	$(PY) -m pytest -q

report: venv
	$(PY) -m judge.report

check: venv
	$(PY) -m judge.report --check

reproduce: test check

run: venv
	$(PY) -m judge.run --version v1 --split dev
	$(PY) -m judge.run --version v2 --split dev
	$(PY) -m judge.run --version v1 --gaming
	$(PY) -m judge.run --version v2 --gaming
	$(PY) -m judge.run --final

data: venv
	mkdir -p .download && curl -L -o .download/complaints.csv.zip $(MIRROR)
	$(PY) -m judge.data --mirror .download/complaints.csv.zip
