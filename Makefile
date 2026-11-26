PYTHON ?= .venv/bin/python
MODELS = lgb_fe lgb_tuned xgb cat rf et lr mlp ord
OOF_FILES = $(addprefix out/oof/, $(addsuffix .npy, $(MODELS)))

.PHONY: all preprocess tune train stack report clean

all: submissions/submission_stack.csv report

data/processed/X_train.parquet data/processed/folds.npy: src/preprocess.py data/raw/dataset_train.csv data/raw/dataset_test_X.csv config.yaml
	$(PYTHON) main.py preprocess

preprocess: data/processed/X_train.parquet

tune: data/processed/X_train.parquet
	$(PYTHON) main.py tune

out/oof/%.npy: data/processed/X_train.parquet src/train.py src/models/*.py
	$(PYTHON) main.py train --model $*

train: $(OOF_FILES)

submissions/submission_stack.csv: $(OOF_FILES) src/stack.py
	$(PYTHON) main.py stack

stack: submissions/submission_stack.csv

report: submissions/submission_stack.csv src/report.py
	$(PYTHON) main.py report --all

clean:
	rm -rf data/processed/* out/oof/* out/preds/* out/parts/* out/report/*
