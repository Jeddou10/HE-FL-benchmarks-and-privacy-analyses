# Privacy Meets Performance — CKKS in Federated Learning

This repository contains the experimental harness and scripts used in the bachelor thesis *Privacy Meets Performance: Rethinking Homomorphic Encryption in Federated Learning*. The code implements reproducible benchmarks and privacy attacks for CKKS homomorphic encryption (TenSEAL) applied to federated learning on MNIST (and example extensions for other datasets). The goal is to measure runtime, memory and communication overheads, evaluate engineering optimizations, and quantify privacy leakage under several threat models.

---

## Repository layout

- `fl_ckks_mnist_benchmark.py`  
- `fl_ckks_mnist_benchmark_batching.py`  
- `fl_ckks_mnist_benchmark_quant.py`  
- `fl_ckks_mnist_benchmark_dp.py`  
- `fl_ckks_threat_model_full.py`  
- `fl_ckks_mnist_dp_privacy_tests.py`  
- `plots.py` — helper script to visualize metrics from `metrics.csv`.  
- `requirements.txt` — Python dependencies.  
- `Dockerfile` — reproducibility container.  
- `README.md` — this document.  

---

## Quick overview of scripts

- **Benchmarking scripts**  
  - `fl_ckks_mnist_benchmark.py`: Baseline CKKS performance harness.  
  - `fl_ckks_mnist_benchmark_batching.py`: Adds client encryption-batching.  
  - `fl_ckks_mnist_benchmark_quant.py`: Implements client-side quantization.  
  - `fl_ckks_mnist_benchmark_dp.py`: Integrates CKKS with Differential Privacy.  

- **Privacy & attack harnesses**  
  - `fl_ckks_threat_model_full.py`: Threat-model harness with inversion, membership inference, probing, collusion, timing leakage.  
  - `fl_ckks_mnist_dp_privacy_tests.py`: Same attack suite adapted to the CKKS+DP pipeline.  

---

## Minimum environment & requirements

Recommended Python: `3.10` (works with `python:3.10-slim` Docker base). Example pinned packages are in `requirements.txt`.  
Minimum dependencies:
- `torch`, `torchvision`  
- `tenseal`  
- `numpy`, `pandas`  
- `psutil`, `Pillow`  

**Important:** TenSEAL requires a C++17 toolchain (`cmake`, `g++`, `libomp-dev`). If installation fails locally, build and run via the provided Dockerfile.

---

## Quick start — local (no Docker)

1. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

2. Install requirements:
   ```bash
   pip install -r requirements.txt

3. Run a simple benchmark:
   ```bash
   python fl_ckks_mnist_benchmark.py --clients 10 --rounds 5 --iid

4. Visualize results (accuracy curve):
   ```bash
   python plot_metrics.py --csv ./runs/ckks_mnist/metrics.csv --metric test_acc --out ./curves/acc_curve.png

## Quick start — with Docker
   
1. Build the Docker image:
   ```bash
   docker build -t fl-ckks-thesis .

2. Run a benchmark inside the container:
   ```bash
   docker run --rm -v $(pwd)/runs:/workspace/runs fl-ckks-thesis \
   fl_ckks_mnist_benchmark.py --clients 10 --rounds 5 --iid --outdir /workspace/runs/test

3. Results and logs will be available under ./runs/test.

## Example commands for each script
1. Baseline benchmark
   ```bash
   python fl_ckks_mnist_benchmark.py \
   --clients 50 --rounds 20 --iid \
   --outdir runs/ckks_baseline
   
2. Batching optimization
   ```bash
   python fl_ckks_mnist_benchmark_batching.py \
   --clients 50 --rounds 20 --iid \
   --outdir runs/ckks_batching

3. Quantization optimization
   ```bash
   python fl_ckks_mnist_benchmark_quant.py \
   --clients 50 --rounds 20 --iid \
   --quant-bits 8 --outdir runs/ckks_quant

4. With Differential Privacy


Client DP (local noise):
   ```bash
   python fl_ckks_mnist_benchmark_dp.py \
   --clients 50 --rounds 20 --iid \
   --dp-mode client --dp-clip 1.0 --dp-sigma 0.5 \
   --outdir runs/ckks_dp_client
   ```
Server DP (encrypted noise at server):
   ```bash
   python fl_ckks_mnist_benchmark_dp.py \
   --clients 50 --rounds 20 --iid \
   --dp-mode server --dp-clip 1.0 --dp-sigma 0.5 \
   --outdir runs/ckks_dp_server
   ```

5. Full threat-model harness
   ```bash
   python fl_ckks_threat_model_full.py \
   --clients 8 --rounds 5 --iid \
   --attack-round 3 \
   --outdir runs/threat_model

6. DP + Privacy tests
   ```bash 
   python fl_ckks_mnist_dp_privacy_tests.py \
   --clients 8 --rounds 5 --iid \
   --dp-mode client --dp-clip 1.0 --dp-sigma 0.5 \
   --attack-round 3 \
   --outdir runs/dp_privacy
   ```
## Plotting results

The helper script `plot_metrics.py` can visualize metrics from one or more `metrics.csv` files.

### Example: single run (test accuracy)
    
     python plot_metrics.py --csv runs/ckks_mnist/metrics.csv --metric test_acc
     
### Example: multiple run (test accuracy)
   
     python plot_metrics.py --csv runs/ckks_mnist/metrics.csv runs/ckks_mnist_quant/metrics.csv --metric test_acc
    
### Example: save the plot in a file (test accuracy)
     python plot_metrics.py --csv ./runs/ckks_mnist/metrics.csv --metric test_acc --out ./curves/acc_curve.png
## Outputs

Each performance benchmark produces:

`metrics.csv` — machine-readable log of performance and accuracy.

`metrics.jsonl` — JSON log for round-level metadata.

Each privacy benchmark produces:

`.json` — JSON log for round-level metadata.

`original.png` — original image.

`inversion.png` — reconstructed image using inversion attack.

## Reproducibility & citation

All experiments in the thesis can be replicated using the provided scripts and Docker environment. To cite or reference this work, please refer to the bachelor thesis:

Privacy Meets Performance: Rethinking Homomorphic Encryption in Federated Learning, Oussama Jeddou, 2025.