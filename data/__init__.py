# -------------------------------------------------------
# Dataset bundle code
# -------------------------------------------------------

from omegaconf import DictConfig
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .sudoku import split_sudoku
from .tiny_gsm import split_tinygsm


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    tokenizer: Optional[AutoTokenizer] = None


def setup_data_bundle(config: DictConfig) -> DatasetBundle:
    """
    get the dataset config and return the dataset bundle
    """
    tokenizer = None

    if config.dataset == "sudoku":
        train_data, val_data = split_sudoku(
            config.data_dir,
            config.sudoku_type,
            val_ratio=config.val_ratio,
            seed=config.seed,
            mmap=config.mmap,
            single_seq_sudoku=getattr(config, "single_seq_sudoku", False),
        )
    elif config.dataset == "tinygsm":
        train_data, val_data = split_tinygsm(config.data_dir, val_ratio=config.val_ratio, seed=config.seed)
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.training.cpus,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.training.cpus,
    )
    return DatasetBundle(train_loader, val_loader, tokenizer)


if __name__ == "__main__":
    # sudoku test code
    config = DictConfig({
        "dataset": "sudoku",
        "sudoku_type": "new",
        "data_dir": "data/sudoku_new",
        "val_ratio": 0.05,
        "seed": 2025,
        "mmap": False,
        "training": {
            "per_gpu_batch_size": 16,
            "cpus": 16,
        }
    })
    bundle = setup_data_bundle(config)
    print(bundle)
