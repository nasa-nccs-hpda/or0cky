# ROCKE-3D JAX GPU Benchmarking Guide

This guide provides instructions for running the **ROCKE-3D JAX port** on
**NVIDIA GPUs**. The table below shows *pre-GPU estimates* only — for real
measured numbers (kernel-level: 2.3–6.3×; full physics chain: ~1.0×, i.e.
essentially no GPU benefit at this grid size), see `STATUS.md`.

---

## 🚀 Quick Start (Docker)

### **1. Build the GPU Docker Image**
```bash
cd /home/gtamkin/_ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax
docker build -t rocke3d-jax-gpu -f Dockerfile.gpu .
```

### **2. Run the Container with GPU Access**
```bash
docker run --gpus all -it rocke3d-jax-gpu
```

### **3. Execute Benchmarks Inside Container**
```bash
cd /workspace
python mantle/benchmark_all.py
```

---

## 📦 Manual Setup (Bare Metal)

### **Prerequisites**
- **NVIDIA GPU** (e.g., A100, V100, RTX 3090/4090)
- **CUDA 12.1** (or compatible version)
- **NVIDIA Driver** (≥ 525.60.13)
- **Python 3.9–3.11**

### **1. Install CUDA Toolkit**
Follow NVIDIA’s instructions for your OS:
- [CUDA 12.1 Download](https://developer.nvidia.com/cuda-12-1-0-download-archive)

### **2. Install JAX with GPU Support**
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### **3. Verify GPU Detection**
```bash
python -c "import jax; print('Devices:', jax.devices())"
```
**Expected Output:**
```
Devices: [GpuDevice(id=0, process_index=0, ...)]
```

---

## 📊 GPU Performance (real, measured — see `STATUS.md` for detail)

| **Scope** | **Fortran (CPU)** | **JAX (CPU)** | **JAX (GPU, real A100)** |
|------------|---------------|--------------------|------------------------|
| DRYCNV kernel, P2SAoM40 grid | 1.27 ms | 17.1 ms | 0.56 ms — **2.3× faster than Fortran** |
| PBL kernel, P2SAoM40 grid | 0.37 ms | 0.20 ms | 0.06 ms — **6.3× faster than Fortran** |
| Full physics chain (PBL+SURFACE+GROUND) | 264.0 ms | 33.51 ms | 32.96 ms — **~1.0× vs JAX-CPU, no GPU benefit** |

---

## 🏗️ HPC Cluster Instructions (Slurm)

### **1. Load Required Modules**
```bash
module load cuda/12.1
module load gcc/12.1.0
module load python/3.9
```

### **2. Submit GPU Job**
Create a Slurm script (`run_gpu_benchmark.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=rocke3d-jax-gpu
#SBATCH --output=benchmark_gpu_%j.out
#SBATCH --error=benchmark_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

cd /home/gtamkin/_ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax

# Load modules
module load cuda/12.1
module load gcc/12.1.0
module load python/3.9

# Install JAX with GPU support (if not pre-installed)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Run benchmarks
python mantle/benchmark_all.py > benchmark_gpu_results.txt 2>&1
```

### **3. Submit the Job**
```bash
sbatch run_gpu_benchmark.slurm
```

### **4. Check Results**
```bash
tail -f benchmark_gpu_<JOB_ID>.out
```

---

## 🔧 Troubleshooting

### **1. No GPUs Detected**
**Error:**
```
RuntimeError: Unable to initialize backend 'cuda': FAILED_PRECONDITION: No visible GPU devices.
```
**Solution:**
- Ensure **NVIDIA drivers** are installed (`nvidia-smi` should work).
- Use `JAX_PLATFORMS=cpu` to fall back to CPU.
- For Docker, use `--gpus all` flag.

### **2. CUDA Version Mismatch**
**Error:**
```
Could not load dynamic library 'cudart64_12.dll'
```
**Solution:**
- Install **CUDA 12.1** (match JAX version).
- Check compatibility: [JAX CUDA Versions](https://github.com/google/jax#pip-installation-gpu-cuda)

### **3. Out of Memory (OOM)**
**Error:**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce **grid size** in `mantle/benchmark_all.py`.
- Use **smaller batch sizes** for large modules (e.g., `DRYCNV` with `LM=10`).

---

## 📝 Notes

- **GPU Speedup**: real, measured, and scope-dependent — 2.3–6.3× on isolated
  kernels, ~1.0× (no benefit) on the full chained physics group at this grid
  size. See `STATUS.md`, "Full Physics Chain" section, for why.
- **Multi-GPU**: For larger models, use `jax.distributed` for multi-GPU scaling.
- **TPU Support**: For Google TPUs, use `JAX_PLATFORMS=tpu` and follow [JAX TPU Guide](https://github.com/google/jax#google-tpu).

---

## 📚 References

- [JAX GPU Installation](https://github.com/google/jax#pip-installation-gpu-cuda)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [CUDA 12.1 Documentation](https://docs.nvidia.com/cuda/12.1.1/)
