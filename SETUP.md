# Environment Setup

## Requirements

- Anaconda or Miniconda
- NVIDIA GPU with CUDA 13.0 driver support (RTX 4090 / 5090 recommended)
  - Driver version ≥ 525 (check with `nvidia-smi`)
- `mengine` simulator (provided separately, see step 3)

---

## Step 1 – Create the conda environment

```bash
conda env create -f environment.yml
```

This creates the `16762` environment with Python 3.11, PyTorch (CUDA 13.0),
Tianshou 2.0, Gymnasium, and all other dependencies.

> **If you get a solver error**, create the env manually:
> ```bash
> conda create -n 16762 python=3.11 -y
> conda activate 16762
> pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
> pip install pybullet==3.2.7 screeninfo==0.6.1 scipy numpy
> pip install tianshou==2.0.0 sensai-utils arch rliable
> pip install gymnasium==1.2.3
> pip install wandb tensorboard
> pip install matplotlib pillow
> ```

---

## Step 2 – Activate the environment

```bash
conda activate 16762
```

---

## Step 3 – Install mengine (robot simulator)

`mengine` is the course simulator and must be installed separately from its
local source tree (it is not on PyPI).

```bash
pip install -e /home/ye/mengine
# or, if cloning fresh:
# git clone <mengine-repo-url> ~/mengine
# pip install -e ~/mengine
```

Verify:
```bash
python -c "import mengine; print(mengine.directory)"
```

---

## Step 4 – Log in to Weights & Biases

```bash
wandb login
```

Paste your API key from https://wandb.ai/authorize.
Only needed once per machine.

---

## Step 5 – Run training

```bash
cd /home/ye/16762-project/grasp
python train_grasp.py --project 16762-rl --name GraspPhase1
```

Useful flags:
| Flag | Default | Description |
|---|---|---|
| `--project` | `16762-robot-rl` | W&B project name |
| `--name` | `GraspPhase1` | W&B run name and checkpoint folder |
| `--save-every` | `10` | Save a dated snapshot every N epochs |
| `--record-video` | off | Upload evaluation video to W&B |
| `--video-every` | `10` | Video upload frequency (epochs) |

---

## Verify GPU is visible

```bash
conda run -n 16762 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output example:
```
True NVIDIA GeForce RTX 4090
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA initialization: driver too old` | Update NVIDIA driver to ≥ 525 |
| `FileExistsError: log/GraspPhase1/...` | Normal on re-run; training uses `raise_error_on_dirname_collision=False` automatically |
| `ImportError: mengine` | Run `pip install -e ~/mengine` inside the conda env |
| W&B curves not appearing | Check you are in the correct project; curves appear after the first epoch completes |
