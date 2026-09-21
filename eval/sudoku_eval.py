import torch
import os
import numpy as np
import math
import torch.distributed as dist
from sampling import mdm_sampling
from tqdm import tqdm

# sudoku eval helper

def evaluate_ddp_sudoku(model, cfg, device, rank: int, world_size: int, sampling):
    val_dir = cfg.validation.val_dir
    mask_id = cfg.data.mask_id

    # The cache stores [puzzle (81), solution (81)], while sampling always
    # operates in place on one 81-token puzzle grid.
    raw = np.load(os.path.join(val_dir, "test_mdm.npy"))
    if raw.ndim != 2 or raw.shape[1] != 162:
        raise ValueError(
            "Sudoku cache must have shape [N, 162] with "
            f"[puzzle (81), solution (81)] records; got {raw.shape}"
        )
    puzzle = raw[:, :81].copy()
    X = np.where(puzzle == 0, mask_id, puzzle)

    N = len(X)
    # for our initial runs, we split the validation set (time efficiency)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X = X[:N_val]
    puzzle = puzzle[:N_val]

    # distribute test cases
    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)

    batch_size = 16
    num_batches = math.ceil((end - start) / batch_size)
    local_correct, local_total = 0, 0

    with torch.no_grad():
        for j in tqdm(range(num_batches), desc = "Evaluating"):
            s = start + j * batch_size
            e = min(s + batch_size, end)
            batch_X = torch.from_numpy(X[s:e]).long().to(device)
            batch_puzzle = torch.from_numpy(puzzle[s:e]).long().to(device)

            pred = mdm_sampling(model, batch_X, mask_id, sampling, device)
            matches = verify_sudoku(pred, batch_puzzle)
            local_correct += matches.sum().item()
            local_total += batch_X.shape[0]

    # accumulate succcess rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()

    return global_correct / global_total

def verify_sudoku(pred: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, 81] predicted solution
    puzzle: [B, 81] original puzzle (0 for empty, 1-9 for clues)
    returns: [B] bool
    """
    clue_ok = ((puzzle == 0) | (pred == puzzle)).all(dim=1)
    sudoku_ok = sudoku_check(pred)
    return clue_ok & sudoku_ok

def sudoku_check(pred: torch.Tensor) -> torch.Tensor:
    """
    Check if the predicted Sudoku solution is valid.
    pred: [B, 81], returns [B] bool
    """
    B, _ = pred.shape
    x = pred.view(B, 9, 9)

    # Must be integers in {1,...,9} (no zeros allowed in a completed Sudoku)
    in_range = (x >= 1) & (x <= 9)

    # Helper: check each length-9 group is a permutation of 1..9
    ref = torch.arange(1, 10, device=pred.device, dtype=pred.dtype).view(1, 1, 9)

    def groups_ok(groups: torch.Tensor) -> torch.Tensor:
        # groups: [B, G, 9]
        sorted_groups, _ = torch.sort(groups, dim=-1)
        return (sorted_groups == ref).all(dim=-1)  # [B, G] bool

    # Rows: [B, 9, 9]
    rows_ok = groups_ok(x)

    # Cols: [B, 9, 9]
    cols_ok = groups_ok(x.transpose(1, 2))

    # 3x3 blocks: reshape into 9 blocks of 9
    blocks = x.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).contiguous().view(B, 9, 9)
    blocks_ok = groups_ok(blocks)

    # All constraints must hold + all entries in range
    return in_range.all(dim=(1, 2)) & rows_ok.all(dim=1) & cols_ok.all(dim=1) & blocks_ok.all(dim=1)
