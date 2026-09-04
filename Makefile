# Oculomics — common entry points. `make help` lists them.
PY ?= python3           # macOS: make setup PY=/usr/bin/python3 (3.9); default python3 may be 3.14
SEEDS ?= 0 1 2 3 4
CKPT ?= ck_kd_v4_384/kd_seed1.pt
B ?=
M ?=

.PHONY: help setup check test smoke reproduce stats deploy export ophthalmic clean-scratch

.DEFAULT_GOAL := help

help:  ## list targets
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | sed "s/:.*## /  -  /"

setup:  ## create .venv (PY=/usr/bin/python3 on macOS) and install pinned deps
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip wheel
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python check_env.py

check:  ## environment + deploy toolchain probe
	.venv/bin/python check_env.py --deploy

test:  ## CPU-only unit tests (~10 s)
	.venv/bin/python -m pytest

smoke:  ## 1-epoch GCG-vs-control harness run into scratch dirs (never clobbers checkpoints/)
	.venv/bin/python run_experiment.py --seeds 0 --quick --epochs 1 --out-dir experiments_quick --ckpt-dir checkpoints_quick

reproduce:  ## headline BRSET->mBRSET result: make reproduce B=... M=... [SEEDS="0 1 2"]
	B="$(B)" M="$(M)" SEEDS="$(SEEDS)" bash reproduce.sh

stats:  ## re-summarise an existing sweep
	B="$(B)" M="$(M)" STAGE=stats bash reproduce.sh

deploy:  ## operating point + Core ML for CKPT (default ck_kd_v4_384/kd_seed1.pt)
	B="$(B)" M="$(M)" STAGE=deploy DEPLOY_CK="$(CKPT)" bash reproduce.sh

ophthalmic:  ## BRSET ophthalmic labels: multi-label head vs one model per label: make ophthalmic B=... [SEEDS="0 1 2"]
	B="$(B)" bash run_ophthalmic.sh $(SEEDS)

export:  ## Core ML export only: make export CKPT=path.pt
	.venv/bin/python export_coreml.py --checkpoint "$(CKPT)"

clean-scratch:  ## remove smoke-test outputs and caches
	rm -rf experiments_quick checkpoints_quick .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
