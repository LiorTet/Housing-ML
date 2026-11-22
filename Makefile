.PHONY: install profile data train register gate approve rollback pipeline leaderboard evaluate drift traffic serve ui mlflow test lint format smoke clean

ENV ?= dev
# uv-managed Python: same interpreter (with lzma) on every machine.
export UV_PYTHON_PREFERENCE := only-managed
PY = uv run python
CLI = $(PY) -m housing --env $(ENV)
EXPERIMENT ?= experiments/baseline.yaml
TRIGGER ?= manual

install:
	uv sync --locked --extra ui

profile:
	uv run --group analysis jupyter nbconvert --to notebook --execute --inplace analysis/data_profile.ipynb

data:
	$(CLI) data

train:
	$(CLI) train --experiment $(EXPERIMENT) --trigger $(TRIGGER)

register:
	$(CLI) register --run $(RUN)

gate:
	$(CLI) gate $(if $(VERSION),--version $(VERSION))

pipeline: data train gate

approve:
	$(CLI) approve $(if $(VERSION),--version $(VERSION))

rollback:
	$(CLI) rollback

leaderboard:
	$(CLI) leaderboard

evaluate:
	$(CLI) evaluate --data $(DATA)

drift:
	$(CLI) drift

traffic:
	$(CLI) traffic --n $(or $(N),200) --shift $(or $(SHIFT),1.0)

serve:
	ENV=$(ENV) $(PY) -m uvicorn housing.api:create_app --factory --port 8000

ui:
	uv run --extra ui python -m streamlit run src/housing/ui.py

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow/$(ENV).db --port 5000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

smoke:
	$(MAKE) pipeline ENV=dev

clean:
	rm -rf data/dev data/prod mlflow logs
