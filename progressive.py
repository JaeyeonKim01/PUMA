from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
import torch
import math

# -----------------------------------------
# helper function
# -----------------------------------------

def mdm_loss_fn(log_probs: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, mask_id: int, prompt_mask: torch.Tensor, arm_init: bool = False) -> torch.Tensor:
    """
    compute the MDM (reweighted) loss with the given probs 
    the progessive masking strategy requires log probs, so we cannot use the CE loss directly
    """
    B, L, V = log_probs.shape
    masked = (xt == mask_id)

    if arm_init:
        masked = masked[:, 1:]
        x0 = x0[:, 1:]
        log_probs = log_probs[:, :-1, :]
        prompt_mask = prompt_mask[:, 1:]
        L = L - 1

    L_eff = L - prompt_mask.sum(dim=1, keepdim=True)
    num_mask = masked.sum(dim=1, keepdim=True).clamp_min(1).float()

    if masked.sum().item() == 0:
        return log_probs.sum() * 0.0
    
    # compute the likelihood w.r.t. the true positions
    nll = -log_probs.gather(dim = -1, index = x0.unsqueeze(-1)).squeeze(-1)
    per_seq_loss = (nll * masked).sum(dim = 1, keepdim=True)

    # calculate the weights per seq
    return (per_seq_loss / num_mask).sum() / B

def build_intervals(K : int) -> List[Tuple[float, float]]:
    """
    Build K *ratio* intervals spanning [0, 1].
    Each interval is [j/K, (j+1)/K], and we will sample ratio ~ Uniform([lo, hi)).
    """
    if K <= 0:
        return []
    return [(j / K, (j + 1) / K) for j in range(K)]


def phase_initialize(B: int, K: int, device: torch.device) -> torch.Tensor:
    """
    initialize the phase tensor
    """
    base = torch.arange(K, device=device)
    repeats = math.ceil(B / K)
    phases = base.repeat(repeats)[:B]
    phase = phases[torch.randperm(B, device=device)]
    return phase


def unmask_from_scores(scores: torch.Tensor, num_unmask: torch.Tensor, x0: torch.Tensor, xt_format: torch.Tensor) -> torch.Tensor:
    """
    unmask the tokens from the scores, both used at the batch update & refill
    """
    B = scores.shape[0]
    k_max = int(num_unmask.max().item())
    new_xt = xt_format.clone()
    if k_max > 0:
        _, topk_idx = scores.topk(k = k_max, dim = 1, largest = True) # [B, k_max]
        arange_k = torch.arange(k_max, device=scores.device).unsqueeze(0).expand(B, k_max) # [B, k_max], indexing helper
        unmask_idx = (arange_k < num_unmask.unsqueeze(1)) # the actual indices to unmask

        # get the unmasking positions
        rows = torch.arange(B, device=scores.device).unsqueeze(1).expand(B, k_max) # [B, k_max]
        flat_r = rows[unmask_idx] # [B*k_max]
        flat_c = topk_idx[unmask_idx] # [B*k_max]
        new_xt[flat_r, flat_c] = x0[flat_r, flat_c]

    return new_xt

# -----------------------------------------
# phased masking class
# -----------------------------------------

class PhasedMasking:
    """
    our progressive unmasking strategy:
        * sample target (#unmasked ratio) from the next interval
        * convert the ratio to the number of unmasked tokens as 
            num_unmask_i = round(L_eff * ratio_i)
        * reveal the top-k tokens with the highest logits
        * if a seq goes through all stages, we refill it with a new sequence from the train loader
    """

    def __init__(self, train_loader: DataLoader, batch_size: int, mask_id: int, K: int, device: torch.device, L: int, mode: str = "standard", confidence_threshold: Optional[float] = None, eos_id: Optional[int] = None):
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.K = K
        self.device = device
        self.L = L
        self.mode = mode
        self.eos_id = eos_id
        self.confidence_threshold = confidence_threshold
        assert mode in ["standard" , "confidence_collapse"], "invalid/deprecated mode"

        # build intervals
        self.intervals = build_intervals(K)

        # cache interval values as tensors
        self.lower = torch.tensor([a for (a,b) in self.intervals] , device=device , dtype=torch.float)
        self.upper = torch.tensor([b for (a,b) in self.intervals] , device=device , dtype=torch.float)

        # state: variable used for training
        B = self.batch_size
        self.state = dict(
            t = 0, # time step
            phase = phase_initialize(B, K, device),
            prompt_mask = torch.zeros(B, L, dtype = torch.bool, device=device), # prompt mask
            L_eff = torch.zeros(B, dtype = torch.long, device=device), # per-seq effective length
            eos_mask = torch.zeros(B, L, dtype = torch.bool, device=device), # EOS mask
            L_eos = torch.zeros(B, dtype = torch.long, device=device) # per-seq EOS length
        )

        self._reset_iter()
        self._initialize_pool() # very first training batch

    def _reset_iter(self):
        self._iter = iter(self.train_loader)

    def reset_loader_iter(self):
        self._reset_iter()
    
    def current_batch(self) -> torch.Tensor:
        return self.xt
    
    def update_k(self, new_k: int):
        """
        update K and corresponding intervals.
        """
        self.K = new_k
        self.intervals = build_intervals(new_k)

        # cache interval values as tensors
        self.lower = torch.tensor([a for (a,b) in self.intervals] , device=self.device , dtype=torch.float)
        self.upper = torch.tensor([b for (a,b) in self.intervals] , device=self.device , dtype=torch.float)

    def _sample_ratio(self, stages: torch.Tensor) -> torch.Tensor:
        """
        sample the ratio from the next interval
        return: [n] float in [lo, hi)
        """
        lo = self.lower.index_select(0, stages)  # [n] float
        hi = self.upper.index_select(0, stages)  # [n] float
        ratio = lo + torch.rand_like(lo) * (hi - lo)   # [n] float in [lo, hi)
        return ratio

    def _sample_target_unmasked(self, ratio: torch.Tensor, L_eff: torch.Tensor) -> torch.Tensor:
        """
        sample the target #unmasked ratio from the next interval
        return: [n] long, the number of unmasked tokens
        """
        num_unmask = torch.round(ratio * L_eff.float()).long()  # [n]
        num_unmask = torch.minimum(num_unmask, (L_eff - 1).clamp_min(1)) # [n]
        return num_unmask

    def calculate_phase(self, xt: torch.Tensor) -> int:
        """
        calculate the phase based on the current xt
        return: phase (int)
        """
        current_unmask = (~ self.state['prompt_mask'] & (xt != self.mask_id)).sum(dim=1).long()
        ratio = current_unmask.float() / self.state['L_eff'].clamp_min(1).float()
        boundaries = self.upper[ : -1]
        stage = torch.bucketize(ratio, boundaries)
        return stage.clamp_(0, self.K - 1).long()
        

    @torch.no_grad()
    def _refill_pool(self, n: int, stages: torch.Tensor):
        """
        initialize the pool of sequences
         * used at the very first training batch and refill
         * for the first step, n = B, for the refill, n = n_new
        """
        L, device = self.L, self.device
        new_x0, new_masks = self._get_new_seq(n)

        # per-seq effective length (# non-prompt tokens)
        new_L_eff = (~new_masks).sum(dim = 1).long() # [n]
        ratio = self._sample_ratio(stages) # [n]
        u0 = self._sample_target_unmasked(ratio, new_L_eff) # [n]

        # contstruct xt, while maintaining prompts
        new_xt = torch.full_like(new_x0, self.mask_id)
        new_xt = torch.where(new_masks, new_x0, new_xt)
        k_max = int(u0.max().item())
        if k_max > 0:
            # the score is random for the first step, setting -inf for prompt positions
            rand_score = torch.rand(n , L, device=device, dtype=torch.float)
            rand_score = torch.where(new_masks, torch.finfo(rand_score.dtype).min, rand_score)
            new_xt = unmask_from_scores(rand_score, u0, new_x0, new_xt)

        return new_x0, new_xt, new_masks, new_L_eff

    @torch.no_grad()
    def _initialize_pool(self):
        """
        initialize the pool of sequences
        """
        B = self.batch_size
        phases = self.state['phase']
        new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(B, phases)
        self.x0 = new_x0
        self.xt = new_xt
        self.state['prompt_mask'] = new_masks
        self.state['L_eff'] = new_L_eff


    @torch.no_grad()
    def _get_new_seq(self, n : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get new sequences from the trainloader, 
        returns: x0 (n, L), prompt_mask (n, L)
        """
        out, masks = [], []
        while len(out) < n:
            try:
                batch = next(self._iter)
            except StopIteration:
                self._reset_iter()
                batch = next(self._iter)
            labels, prompt_mask = batch['labels'], batch['prompt_mask']
            if labels.ndim == 1:
                out.append(labels)
                masks.append(prompt_mask)
            else:
                for t, m in zip(labels, prompt_mask):
                    out.append(t)
                    masks.append(m)
                    if len(out) >= n:
                        break
        return torch.stack(out , dim = 0).to(self.device), torch.stack(masks , dim = 0).to(self.device)

    @torch.no_grad()
    def update_with_logits(self, log_probs: torch.Tensor):
        """
        build the next batch of sequences following our method
        """
        B, L, V = log_probs.shape
        device = self.device

        # stages for the current update
        phase_next = (self.state['phase'] + 1) % self.K
        replace = (phase_next == 0)

        mask_idx = (self.xt == self.mask_id) # [B, L]
        assert not (mask_idx & self.state['prompt_mask']).any(), "prompt positions should not be masked"

        ratio = self._sample_ratio(phase_next) # [B]
        num_unmask = self._sample_target_unmasked(ratio, self.state['L_eff'])
        current_num_unmask = (~mask_idx & ~self.state['prompt_mask']).sum(dim=1).long()
        to_reveal = (num_unmask - current_num_unmask).clamp_min(0) # the number of actual tokens to reveal

        # don't update if replace == True
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        # progressive unmasking
        k_max = int(to_reveal.max().item())
        xt = self.xt
        if k_max > 0:
            score_conf = torch.where(mask_idx , log_probs.max(dim = 2)[0], torch.finfo(log_probs.dtype).min) # [B, L]
            xt = unmask_from_scores(score_conf, to_reveal, self.x0, self.xt)

        if self.mode in ["confidence_collapse"]:
            tau = math.log(self.confidence_threshold)
            p = log_probs.max(dim = 2)[0]
            update_unmask = (p > tau) & (xt == self.mask_id) & (~ self.state['prompt_mask'])
            xt = torch.where(update_unmask, self.x0, xt)

            # re-calculate the phase
            phase_next = self.calculate_phase(xt)

        self.xt = xt
        self.state['phase'] = phase_next

        
        # if a seq goes through all stages, we refill
        n_new = int(replace.sum().item())
        if n_new > 0:
            idx = replace.nonzero(as_tuple = False).squeeze(1) # [n_new]
            stages = torch.zeros(n_new, device=device, dtype=torch.long)
            new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(n_new, stages)
            self.x0[idx] = new_x0
            self.xt[idx] = new_xt
            self.state['prompt_mask'][idx] = new_masks
            self.state['L_eff'][idx] = new_L_eff
            self.state['phase'][idx] = 0

        self.state['t'] += 1 # update the time step
