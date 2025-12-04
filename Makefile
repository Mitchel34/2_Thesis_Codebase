PYTHON ?= python3
PYTHONPATH ?= .
DATA_PATH ?= data/clean/modeling/hourly_training_all_sites_20100101_20201231.parquet
OUTPUT_PREFIX ?= hydra_v2_full_q
FIGURES_DIR ?= results/figures/$(OUTPUT_PREFIX)
STATION_NAME ?= "Watauga River, NC"
HPO_STUDY ?= hydra_optuna
HPO_TRIALS ?= 8
HPO_OUTPUT_PREFIX ?= hydra_v2_optuna

# Hyperparameters (override via command line when needed)
SEQ_LEN ?= 168
EPOCHS ?= 40
BATCH_SIZE ?= 64
TRAIN_START ?= 2010-01-01
TRAIN_END ?= 2018-12-31
VAL_START ?= 2019-01-01
VAL_END ?= 2019-12-31
TRAIN_DAYS ?= 3287
VAL_DAYS ?= 365
PATIENCE ?= 12
D_MODEL ?= 128
NUM_HEADS ?= 4
NUM_LAYERS ?= 4
CONV_DEPTH ?= 4
DROPOUT ?= 0.1
LR ?= 5e-4

TRAIN_ARGS = \
	--data $(DATA_PATH) \
	--seq-len $(SEQ_LEN) \
	--epochs $(EPOCHS) \
	--batch-size $(BATCH_SIZE) \
	--train-days $(TRAIN_DAYS) \
	--val-days $(VAL_DAYS) \
	--train-start $(TRAIN_START) \
	--train-end $(TRAIN_END) \
	--val-start $(VAL_START) \
	--val-end $(VAL_END) \
	--patience $(PATIENCE) \
	--d-model $(D_MODEL) \
	--num-heads $(NUM_HEADS) \
	--num-layers $(NUM_LAYERS) \
	--conv-depth $(CONV_DEPTH) \
	--dropout $(DROPOUT) \
	--lr $(LR) \
	--no-compile \
	--output-prefix $(OUTPUT_PREFIX)

EVAL_CSV = data/clean/modeling/$(OUTPUT_PREFIX)_eval.csv

.PHONY: all train_full plots_full clean_outputs hpo

all: train_full plots_full

train_full: $(EVAL_CSV)

$(EVAL_CSV):
	@echo "Training Hydra v2 with full timeframe splits (train=2010-2018, val=2019, test=2020)"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) modeling/train_quick_transformer_torch.py $(TRAIN_ARGS)
	@echo "Training complete; evaluation written to $(EVAL_CSV)"

plots_full: $(EVAL_CSV)
	@mkdir -p $(FIGURES_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m modeling.plot_suite \
		--model hydra=$(EVAL_CSV) \
		--out-dir $(FIGURES_DIR) \
		--station $(STATION_NAME) \
		--monthly
	@echo "Plots saved to $(FIGURES_DIR)"

hpo:
	@echo "Starting Optuna hyperparameter search with $(HPO_TRIALS) trials"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) modeling/hpo_optuna.py \
		--data $(DATA_PATH) \
		--study-name $(HPO_STUDY) \
		--n-trials $(HPO_TRIALS) \
		--epochs 25 \
		--batch-size 64 \
		--patience 5 \
		--output-prefix $(HPO_OUTPUT_PREFIX)

clean_outputs:
	rm -f data/clean/modeling/$(OUTPUT_PREFIX)_*.npy \
	           data/clean/modeling/$(OUTPUT_PREFIX)_*.csv \
	           data/clean/modeling/$(OUTPUT_PREFIX)_*.json
	rm -rf $(FIGURES_DIR)
