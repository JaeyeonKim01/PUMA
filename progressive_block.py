from progressive import phase_initialize, unmask_from_scores, build_intervals
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import torch
import math
# -----------------------------------------
# phase initialization for block-wise masking
# -----------------------------------------
def phase_initialize_block(B: int, num_blocks: int, K: int, device: torch.device) -> torch.Tensor:
    # phases[b, k] in {0,...,K-1}, shuffled per block
    phases = torch.empty((B, num_blocks), device=device, dtype=torch.long)
    for k in range(num_blocks):
        phases[:, k] = phase_initialize(B, K, device)  # reuse your helper
    return phases

class ProgressiveBlock:
    """
    progressive masking for block diffusion training
        * logic is the same as PhasedMasking, but with block-wise masking
        * K here indicates the number of stages that each block goes through
        # samely, we refill a new seq once a given seq goes through all stages
    """

    def __init__(self, train_loader: DataLoader, batch_size: int, block_size: int, mask_id: int, K: int, device: torch.device, L: int, mode: str = "standard", confidence_threshold: Optional[float] = None, eos_id: Optional[int] = None):
        self.block_size = block_size
        assert L % block_size == 0, "max_len must be divisible by block_size"
        assert mode in ["standard", "confidence_collapse"], "mode must be either standard or confidence_collapse"

        self.train_loader = train_loader
        self.batch_size = batch_size
        self.block_size = block_size
        self.mask_id = mask_id
        self.K = K
        self.device = device
        self.L = L
        self.num_block = L // block_size
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        self.intervals = build_intervals(K)
        self.lower = torch.tensor([a for (a,b) in self.intervals] , device=device , dtype=torch.float)
        self.upper = torch.tensor([b for (a,b) in self.intervals] , device=device , dtype=torch.float)

        self.state = dict(
            t = 0,
            phase = phase_initialize_block(batch_size, self.num_block, K, device),
            done = torch.zeros(batch_size, self.num_block, dtype = torch.bool, device = device),
            prompt_mask = torch.zeros(batch_size, self.L , device = device , dtype = torch.bool),
            L_eff = torch.zeros(batch_size, self.num_block, device = device, dtype = torch.long),
        )

        self._reset_iter()
        self._initialize_pool()

    
    # ------ sampling helpers -------

    def _reset_iter(self):
        self._iter = iter(self.train_loader)

    def reset_loader_iter(self):
        self._reset_iter()

    def _sample_ratio(self, stages_1d: torch.Tensor) -> torch.Tensor:
        lo = self.lower.index_select(0, stages_1d)
        hi = self.upper.index_select(0, stages_1d)
        return lo + torch.rand_like(lo) * (hi - lo)

    def _sample_target_unmasked(self, ratio: torch.Tensor, L_eff: torch.Tensor) -> torch.Tensor:
        # target unmasked count; cap at L_eff-1 (keeps at least 1 masked if possible)
        num_unmask = torch.round(ratio * L_eff.float()).long()
        cap = (L_eff - 1).clamp_min(0)
        num_unmask = torch.minimum(num_unmask, cap)
        num_unmask = num_unmask.clamp_min(0)
        return num_unmask

    @torch.no_grad()
    def _get_new_seq(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out, masks = [], []
        while len(out) < n:
            try:
                batch = next(self._iter)
            except StopIteration:
                self._reset_iter()
                batch = next(self._iter)
            labels, prompt_mask = batch["labels"], batch["prompt_mask"]

            if labels.ndim == 1:
                out.append(labels)
                masks.append(prompt_mask)
            else:
                for t, m in zip(labels, prompt_mask):
                    out.append(t)
                    masks.append(m)
                    if len(out) >= n:
                        break
        x0 = torch.stack(out, dim=0).to(self.device)
        pm = torch.stack(masks, dim=0).to(self.device)
        return x0, pm

    # -------- pool init / refill ---------

    @torch.no_grad()
    def _initialize_pool(self):
        B = self.batch_size
        phases = self.state["phase"]  # (B, nb)
        x0, xt_block, pm, L_eff, done = self._refill_pool(B, phases)

        self.x0 = x0
        self.xt_block = xt_block
        self.state["prompt_mask"] = pm
        self.state["L_eff"] = L_eff
        self.state["done"] = done


    def update_k(self, new_k: int):
        """
        update K and corresponding intervals.
        """
        self.K = new_k
        self.intervals = build_intervals(new_k) 
        self.lower = torch.tensor([a for (a,b) in self.intervals] , device=self.device , dtype=torch.float)
        self.upper = torch.tensor([b for (a,b) in self.intervals] , device=self.device , dtype=torch.float)

        self.state["phase"] %= new_k

    @torch.no_grad()
    def _refill_pool(self, n: int, stages: torch.Tensor):
        """
        stages: (n , num_block) in [0, K-1]
        """
        device = self.device
        nb = self.num_block
        bs = self.block_size

        x0, pm = self._get_new_seq(n)

        # reshape into blocks
        x0_blk = x0.view(n, nb, bs) # (n, nb, bs)
        pm_blk = pm.view(n, nb, bs) # (n, nb, bs)
        L_eff = (~pm_blk).sum(dim = 2).long() # (n, nb)

        # blocks with no valid tokens are done immediately
        done = (L_eff == 0)

        # very initial xt_block: mask everywhere but prompt
        xt_blk = torch.full( (n , nb, bs) , self.mask_id, device = device , dtype = torch.long)
        xt_blk = torch.where(pm_blk, x0_blk, xt_blk)

        # sample target unmasked counts for each
        stages_flat = stages.reshape(-1) # (n * nb,)
        ratio_flat = self._sample_ratio(stages_flat) # (n * nb,)
        L_eff_flat = L_eff.reshape(-1) # (n * nb,)
        u0_flat = self._sample_target_unmasked(ratio_flat, L_eff_flat) # (n * nb,)
        u0_flat = torch.where(L_eff_flat == 0 , torch.zeros_like(u0_flat), u0_flat)

        # randomly unmask u0 tokens within each block among non-prompt positions
        M = n * nb
        scores = torch.rand((M, bs), device = device, dtype = torch.float)
        valid_pos = (~ pm_blk).reshape(M, bs)
        scores = torch.where(valid_pos, scores, torch.finfo(scores.dtype).min)

        xt_flat = xt_blk.reshape(M, bs)
        x0_flat = x0_blk.reshape(M, bs)
        xt_flat = unmask_from_scores(scores, u0_flat, x0_flat, xt_flat)

        xt_blk = xt_flat.view(n, nb, bs)

        return x0, xt_blk, pm, L_eff, done

    # -------- get a model input for a block ---------

    def get_block_batch(self, block_id : int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
            xt_prefix: (B, e)
            x0_suffix: (B, e)
            prompt_prefix: (B, e)
        """
        bs = self.block_size
        s = block_id * bs
        e = s + bs

        xt_prefix = torch.cat([ self.x0[:, : s] , self.xt_block[: , block_id , :] ], dim = 1)
        x0_prefix = self.x0[:, : e]
        prompt_prefix = self.state["prompt_mask"][:, : e]
        return xt_prefix, x0_prefix, prompt_prefix

    # -------- update: block-wise progressive unmasking ---------

    @torch.no_grad()
    def update_block(self, block_id: int, log_probs_prefix: torch.Tensor):
        """
        log_probs_prefix: (B, e, V)
        updates only [: , block_id, :]
        """
        B, e, V = log_probs_prefix.shape
        bs = self.block_size
        s = block_id * bs
        assert e == (s + bs), "block size mismatch"

        pm_blk = self.state["prompt_mask"][: , s : e]
        L_eff_blk = self.state["L_eff"][: , block_id]
        xt_blk = self.xt_block[:, block_id, :]
        x0_blk = self.x0[:, s : e]

        mask_idx = (xt_blk == self.mask_id) # [B, bs] bool
        if (mask_idx & pm_blk).any():
            raise ValueError("prompt positions should not be masked")

        active = (~ self.state["done"][: , block_id]) & (L_eff_blk > 0) # [B] bool

        phase = self.state["phase"][: , block_id]
        phase_next = (phase + 1) % self.K
        done_now = (phase_next == 0) & active # [B] bool

        self.state["done"][: , block_id] |= done_now

        # compute target num unmask
        ratio = self._sample_ratio(phase_next)
        target_unmask = self._sample_target_unmasked(ratio, L_eff_blk)
        current_unmask = ( (~ mask_idx) & (~ pm_blk) ).sum(dim = 1).long()
        to_reveal = (target_unmask - current_unmask).clamp_min(0)

        to_reveal = torch.where(active & (~done_now), to_reveal, torch.zeros_like(to_reveal))

        # confidence scores
        score = log_probs_prefix[: , s:e , :].max(dim = 2)[0]
        score = torch.where(mask_idx  , score, torch.finfo(score.dtype).min)
        score = torch.where(~pm_blk , score, torch.finfo(score.dtype).min)
        
        xt_blk = unmask_from_scores(score, to_reveal, x0_blk, xt_blk)

        if self.mode in ['confidence_collapse']:
            tau = math.log(self.confidence_threshold)
            pmax = log_probs_prefix[: , s:e , :].max(dim = 2)[0]
            collapse = (pmax > tau) & (xt_blk == self.mask_id) & (~ pm_blk) & active[: , None]
            xt_blk = torch.where(collapse, x0_blk, xt_blk)

            # re-calculate the phase
            current_unmask = ( (xt_blk != self.mask_id) & (~ pm_blk) ).sum(dim = 1).long()
            ratio_now = current_unmask.float() / L_eff_blk.clamp_min(1).float()
            boundaries = self.upper[ : -1]
            phase_next = torch.bucketize(ratio_now, boundaries).clamp_(0, self.K - 1).long()

        self.xt_block[:, block_id, :] = xt_blk
        self.state["phase"][: , block_id] = phase_next

    # -------- finish the step --------
    
    @torch.no_grad()
    def finish_step(self):
        self.state["t"] += 1

        done_seq = self.state["done"].all(dim = 1)
        if not done_seq.any():
            return

        idx = done_seq.nonzero(as_tuple = False).squeeze(1) # [n_new]
        n_new = idx.numel()

        stages = torch.zeros( (n_new , self.num_block) , device = self.device, dtype = torch.long)
        x0, xt_block, pm, L_eff, done = self._refill_pool(n_new, stages)

        self.x0[idx] = x0
        self.xt_block[idx] = xt_block
        self.state["prompt_mask"][idx] = pm
        self.state["L_eff"][idx] = L_eff
        self.state["done"][idx] = done
        self.state["phase"][idx] = 0

        
        
        