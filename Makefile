# Makefile for Homomorphic Encryption in Federated Learning experiments

PYTHON      := python3
SCRIPTS_DIR := .
RUNS_DIR    := runs
DATA_DIR    := data

# Scripts
BENCHMARK        := $(SCRIPTS_DIR)/fl_ckks_mnist_benchmark.py
PRIVACY_FULL     := $(SCRIPTS_DIR)/fl_ckks_threat_model_full.py
QUANT_BENCHMARK  := $(SCRIPTS_DIR)/fl_ckks_quant_benchmark.py
BATCH_BENCHMARK  := $(SCRIPTS_DIR)/fl_ckks_batch_benchmark.py
DP_BENCHMARK     := $(SCRIPTS_DIR)/fl_ckks_mnist_dp_benchmark.py
DP_PRIVACY       := $(SCRIPTS_DIR)/fl_ckks_mnist_dp_privacy_tests.py

# Default target
all: benchmark privacy

# Baseline CKKS benchmark
benchmark:
	$(PYTHON) $(BENCHMARK) \
		--dataset mnist \
		--model mlp \
		--clients 10 \
		--participation 0.1 \
		--rounds 5 \
		--output_dir $(RUNS_DIR)/ckks_benchmark_example

# Full privacy harness
privacy:
	$(PYTHON) $(PRIVACY_FULL) \
		--dataset mnist \
		--model mlp \
		--clients 10 \
		--participation 0.1 \
		--rounds 5 \
		--enable_inversion \
		--enable_membership \
		--enable_probing \
		--output_dir $(RUNS_DIR)/privacy_example

# Quantization benchmark
quant:
	$(PYTHON) $(QUANT_BENCHMARK) \
		--dataset mnist \
		--model mlp \
		--rounds 5 \
		--quantize_bits 8 \
		--output_dir $(RUNS_DIR)/quant_example

# Batching benchmark
batch:
	$(PYTHON) $(BATCH_BENCHMARK) \
		--dataset mnist \
		--model mlp \
		--rounds 5 \
		--batching_slots 8192 \
		--output_dir $(RUNS_DIR)/batch_example

# DP + HE benchmark
dp:
	$(PYTHON) $(DP_BENCHMARK) \
		--dataset mnist \
		--model mlp \
		--rounds 5 \
		--dp_mode client \
		--dp_clip 1.0 \
		--dp_sigma 0.5 \
		--output_dir $(RUNS_DIR)/dp_example

# DP + HE privacy tests
dp-privacy:
	$(PYTHON) $(DP_PRIVACY) \
		--dataset mnist \
		--model mlp \
		--rounds 5 \
		--dp_mode client \
		--dp_clip 1.0 \
		--dp_sigma 0.5 \
		--attack_all \
		--output_dir $(RUNS_DIR)/dp_privacy_example

# Clean up outputs
clean:
	rm -rf $(RUNS_DIR)/*

# Build docker container
docker-build:
	docker build -t fl-ckks-repro .

# Run inside container
docker-run:
	docker run --rm -v $(PWD)/$(RUNS_DIR):/workspace/$(RUNS_DIR) fl-ckks-repro \
		bash -lc "$(PYTHON) $(BENCHMARK) --dataset mnist --model mlp --output_dir /workspace/$(RUNS_DIR)/docker_example"

.PHONY: all benchmark privacy quant batch dp dp-privacy clean docker-build docker-run
