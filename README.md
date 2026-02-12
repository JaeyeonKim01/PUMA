<div align="center">

# PUMA: Progressive Unmasking for Accelerated Masked Diffusion Training

</div>

---

## Overview

**PUMA (Progressive Unmasking)** is a simple modification to the forward process of **Masked Diffusion Models (MDMs)**.
Instead of training on randomly masked sequences, PUMA aligns that training- and inference- time masking patterns, thereby focusingon inference-aligned masks and speeding up training. 

---
## Quick Start

### 1. Install Environment

```bash
# Create and activate conda environment
conda env create -f environment.yml
```

The slurm training script (see below) activates the environment automatically. To activate manually, run:

```bash
conda activate puma
```

### 2. Data preparation
We provide PUMA codebases for Sudoku and TinyGSM. 

- Sudoku: Download *sudoku-train-data.npy* and *sudoku-test-data.npy* from [here](https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW) and put them in the `data/sudoku_new` folder. 

- TinyGSM: Run `data/tiny_gsm.py` that gives you files `labels.bin`, `meta.json`, and `prompt_mask.bin` for pretraining in the desired `out_dir` directory.

### 3. Run Training
Submit a job using the SLURM script:

```bash
sbatch job.sh
```

The SLURM script calls `train.py`, which handles the training loop.

### 4. Configuration

Config files are located in `yaml_files/`. Edit these YAML files to adjust:
- Model architecture
- Training hyperparameters  
- Dataset settings
- Logging options

We provide one config each for PUMA and the baseline for the following three settings: Sudoku, TinyGSM (standard), TinyGSM (block diffusion).

### 5. Monitoring

Training logs and checkpoints are saved according to the paths specified in your config file. The training file also logs results to wandb.

---

## Explanation of each files
- `train.py`: unified file that handles the MDM pretraining (includes the vanilla MDM pretraining). Self-includes the evaluation accuracy logging.
- `sampling.py`: sampling for a given MDM
- `progressive.py`': PUMA via batch streaming implementation. The implementation detail can be found in Section 3.2 and Appendix B.1.
- `progressive_block.py`: PUMA implementation for block diffusion. 
- `model`: our Qwen2-style attention implementations
- `eval`: eval util functions for Sudoku / TinyGSM

## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{
    kim2026stoptrainingworstprogressive,
    title={{S}top {T}raining for the {W}orst: {P}rogressive {U}nmasking {A}ccelerates {M}asked {D}iffusion {T}raining},
    author={Jaeyeon Kim and Jonathan Geuter and David Alvarez-Melis and Sham Kakade and Sitan Chen},
    year={2026},
    eprint={2602.10314},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2602.10314}, 
}
```