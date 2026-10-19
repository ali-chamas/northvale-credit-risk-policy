# Northvale Credit: single entry point. Run `make setup && make all`.
SHELL   := /bin/bash
PYTHON  ?= $(shell command -v python3.11 || echo /opt/homebrew/bin/python3.11)
PY      := .venv/bin/python
DECK    := Northvale_case_study
# Private LibreOffice profile so the PDF export never collides with another running soffice.
SOFFICE := soffice -env:UserInstallation=file://$(CURDIR)/.lo_profile --headless

.PHONY: setup data build analysis notebooks docs deck deck-qa test lint app all clean

setup:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	$(PY) -m pip install --quiet --no-deps -e .
	cd deck && npm ci --silent

data:
	bash scripts/download_data.sh

build:
	$(PY) -m northvale.build

analysis:
	$(PY) -m northvale.analysis

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 notebooks/*.ipynb

docs:
	$(PY) -m northvale.render_docs

deck:
	$(PY) -m northvale.deck_content
	cd deck && node build_deck.js deck_spec.json $(DECK).pptx
	cd deck && $(SOFFICE) --convert-to pdf --outdir . $(DECK).pptx >/dev/null
	@test -s deck/$(DECK).pdf && echo "deck: deck/$(DECK).pptx and .pdf written"

# Slide images for visual QA (not committed)
deck-qa:
	mkdir -p deck/qa && pdftoppm -jpeg -r 80 deck/$(DECK).pdf deck/qa/slide

lint:
	$(PY) -m ruff check src tests app

test: lint
	$(PY) -m pytest -q
	$(PY) -m northvale.check_text README.md INTERVIEW_NOTES.md DEPLOY.md deck/deck_spec.json docs/*.md

app:
	$(PY) -m streamlit run app/streamlit_app.py

all: data build analysis notebooks docs deck test

# Generated outputs only. Raw data and the app screenshots (taken by hand) are kept.
clean:
	rm -rf data/interim reports/tables reports/metrics.json deck/qa .lo_profile
	find reports/figures -name '*.png' ! -name 'app_*' -delete
	rm -f app/data/*.parquet app/data/*.json deck/deck_spec.json deck/$(DECK).pptx deck/$(DECK).pdf
