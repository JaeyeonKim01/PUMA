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
    single_seq = getattr(cfg.data, "single_seq_sudoku", False)

    # get the test Sudoku puzzle and answers
    raw = np.load(os.path.join(val_dir, "test_mdm.npy"))
    if single_seq:
        puzzle = raw[:, :81].copy()
        Y = raw[:, 81:].copy()
        X = np.where(puzzle == 0, mask_id, puzzle)
    else:
        X = raw.copy()
        X[:, 81:] = mask_id
        Y = raw.copy()

    N = len(X)
    # for our initial runs, we split the validation set (time efficiency)
    ratio = cfg.validation.ratio
    N_val = int(N * ratio)
    X, Y = X[:N_val], Y[:N_val]
    if single_seq:
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
            batch_Y = torch.from_numpy(Y[s:e]).long().to(device)

            pred = mdm_sampling(model, batch_X, mask_id, sampling, device)
            if single_seq:
                batch_puzzle = torch.from_numpy(puzzle[s:e]).long().to(device)
                matches = verify_sudoku_single(pred, batch_puzzle)
            else:
                matches = verify_sudoku(pred, batch_Y)
            local_correct += matches.sum().item()
            local_total += batch_Y.shape[0]

    # accumulate succcess rates
    tensor = torch.tensor([local_correct, local_total], dtype=torch.long, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_correct, global_total = tensor.tolist()

    return global_correct / global_total

def verify_sudoku_single(pred: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, 81] predicted solution
    puzzle: [B, 81] original puzzle (0 for empty, 1-9 for clues)
    returns: [B] bool
    """
    clue_ok = ((puzzle == 0) | (pred == puzzle)).all(dim=1)
    sudoku_ok = sudoku_check(pred)
    return clue_ok & sudoku_ok

def verify_sudoku(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, 162] where pred[:, :81] are clues/condition, pred[:, 81:] is predicted solution
    target: [B, 162] where target[:, :81] are clues/condition, target[:, 81:] is ground-truth solution
    returns: [B] bool
    """
    cond = pred[:, :81]
    sol  = pred[:, 81:]

    clue_ok = ((cond == 0) | (sol == cond)).all(dim=1)   # [B]
    sudoku_ok = sudoku_check(sol)                        # [B]

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
