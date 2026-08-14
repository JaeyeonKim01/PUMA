# -------------------------------------------------------
# Sudoku data collactor opt for our DataBundle format
# -------------------------------------------------------

import json
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from .preprocess_sudoku import sudoku_example_to_162

class SudokuDataset(Dataset):
    """
    Load cached ``[puzzle (81), solution (81)]`` records and expose one
    81-token model sequence. Puzzle givens are fixed prompt positions; the
    model denoises only cells that were empty in the puzzle.
    """

    def __init__(self, data_dir: str, sudoku_type: str, mmap: bool = True):
        # we sue mmap if the .npy's size is too large
        self.data_dir = data_dir
        
        if sudoku_type == "new":
            # preprocess the sudoku dataset into an MDM-friendly version
            if not os.path.exists(os.path.join(data_dir, "train_mdm.npy")):
                labels = np.load(os.path.join(data_dir, "sudoku-train-data.npy"))
                new_labels = np.zeros((len(labels), 162))
                for i in tqdm(range(len(labels))):
                    new_labels[i], meta = sudoku_example_to_162(labels[i])
                    if not meta["givens_ok"]:
                        print(meta["givens_violations"])
                np.save(os.path.join(data_dir, "train_mdm.npy"), new_labels)
                print("train_mdm.npy saved")
                self.labels = new_labels
            else:
                labels = np.load(os.path.join(data_dir, "train_mdm.npy"))
                self.labels = labels

            # also preprocess for the test dataset
            if not os.path.exists(os.path.join(data_dir, "test_mdm.npy")):
                labels = np.load(os.path.join(data_dir, "sudoku-test-data.npy"))
                new_labels = np.zeros((len(labels) , 162))
                for i in tqdm(range(len(labels))):
                    new_labels[i], meta = sudoku_example_to_162(labels[i])
                    if not meta["givens_ok"]:
                        print(meta["givens_violations"])
                np.save(os.path.join(data_dir, "test_mdm.npy"), new_labels)
                print("test_mdm.npy saved")
        else:
            raise ValueError(f"Invalid sudoku data type: {sudoku_type}")

        if self.labels.ndim != 2 or self.labels.shape[1] != 162:
            raise ValueError(
                "Sudoku cache must have shape [N, 162] with "
                "[puzzle (81), solution (81)] records; "
                f"got {self.labels.shape}"
            )
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # np.load with mmap returns a view, copy() to avoid weirdness
        raw = self.labels[idx].copy()
        puzzle = raw[:81]
        solution = raw[81:]
        lab = torch.from_numpy(solution).long()
        prompt_mask = torch.from_numpy(puzzle != 0).bool()
        return {"labels": lab, "prompt_mask": prompt_mask}


def split_sudoku(
    data_dir: str,
    sudoku_type: str,
    val_ratio: float = 0.05,
    seed: int = 2025,
    mmap: bool = False,
):
    dataset = SudokuDataset(data_dir, sudoku_type, mmap=mmap)
    n = len(dataset)

    # split
    n_val, n_train = int(n * val_ratio), n - int(n * val_ratio)

    # reproducible split
    g = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(dataset, [n_train, n_val], generator=g)
    
    return train_data, val_data
