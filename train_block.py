# -----------------------------------------------------------
# training code for the block diffusion
# NOTE: no KV-caching supported yet.
#       much slower than the full, standard MDM training
# NOTE: we omit the val loss logging
# -----------------------------------------------------------

import math, os, time, json, random, sys, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributed as dist
import argparse
from copy import deepcopy
from tqdm import tqdm
from model.transformer import MDMTransformer, MDMConfig
from data import setup_data_bundle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig
from model.ema import ExponentialMovingAverage, save_ema_snapshot
from progressive import mdm_loss_fn
from progressive_block import ProgressiveBlock
from eval.sudoku_eval import evaluate_ddp_sudoku
from eval.gsm8k_eval import evaluate_ddp_gsm8k
from train import parse_args, setup_ddp, evaluate_ddp_dict, grad_norm, evaluate_ddp,  parse_k_schedule_increasing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    return parser.parse_args()


def mdm_loss_block(model, input_ids, block_size, mask_id, prompt_mask = None):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids , dtype=torch.bool)
    device = input_ids.device
    B, L, L_blk = input_ids.shape[0], input_ids.shape[1], block_size

    assert L % L_blk == 0, "block size must be divisible by the max_length"
    n_blocks = L // L_blk

    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    loss = 0.0
    for n in range(n_blocks):
        s = n * L_blk
        e = s + L_blk
        idx_blk = idx[: , : e]

        valid_ids = (idx_blk >= s) & (~ prompt_mask[: , : e]) # [B, l]
        if valid_ids.sum() == 0:
            # when there's no valid indices in the block
            continue

        L_eff = valid_ids.sum(dim=1)
        valid_seq = (L_eff > 0)
        if not valid_seq.any():
            continue

        # filter seqs with L_eff = 0
        xt = input_ids[valid_seq, : e]
        valid_ = valid_ids[valid_seq]
        L_eff_v = L_eff[valid_seq].float()

        num_mask = (torch.floor(torch.rand(xt.shape[0], device=device) * L_eff_v).long() + 1)
        scores = torch.rand( (xt.shape[0] , e) , device=device).masked_fill(~valid_ , float('inf')).argsort(dim=1)
        order = scores.argsort(dim=1)
        mask_indices = (order < num_mask[: , None])

        # double-check mask_indicies
        assert (mask_indices & ~valid_).sum() == 0, "mask_indices should not contain invalid indices"    

        masked_input = torch.where(mask_indices, mask_id, xt)
        logits = model(masked_input) # [B, e, V]

        ce = F.cross_entropy(logits[mask_indices], xt[mask_indices], reduction="none")
        seq_local = mask_indices.nonzero(as_tuple=False)[:, 0]
        ce = ce / num_mask[seq_local].float()
        loss_blk = ce.sum() / B
        loss_blk.backward()

        loss += loss_blk.item()

    return loss


def main(cfg: DictConfig):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    
    base_seed = 2026
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ckpt dir
    ckpt_dir = f"ckpts/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model_cfg_dict = cfg.model
    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"Model wrapping is done!")

    # data
    data_cfg = cfg.data
    train_cfg = cfg.training
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"
    val_cfg = cfg.validation
    data_bundle = setup_data_bundle(data_cfg)
    train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_training_steps = train_cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    strategy = train_cfg.strategy
    # k schedule for progressive unmasking. If None use fixed K. If "linear", linearly increase the unmasking steps from 1 to K over the training steps.
    # If a list of integers, use the list as the k_steps. If an integer, use constant interval increase.
    if strategy == "progressive":
        k_schedule = parse_k_schedule_increasing(getattr(train_cfg, "k_schedule", None))
        if len(k_schedule) == 0:
            k_schedule = [(train_cfg.K, 0)]
        
        current_k = k_schedule[0][0]

        if is_main:
            print("Using K Schedule:")
            for K, step in k_schedule:
                print(f"Step {step}: K={K}")

        # intialize the pool
        B = train_cfg.batch_size
        L = model_config.max_position
        block_size = train_cfg.block_size
        num_block = L // block_size
        def make_pool(K):
            B = train_cfg.batch_size
            L = model_config.max_position
            return ProgressiveBlock(
                train_loader, B, block_size, mask_id, K, device, L,
                mode=train_cfg.mode,
                confidence_threshold=train_cfg.confidence_threshold,
                eos_id=train_cfg.eos_id,
            )
        pool = make_pool(current_k)
        next_k_idx = 1


    # training loop
    global_step = 0

        
    # wandb initialize
    if cfg.wandb.wandb and is_main:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)
        wandb.termlog("Hey, we start block diffusion training!")
        wandb.termlog(f"Training with {world_size} GPUs")

    for epoch in range(train_cfg.num_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if strategy == "progressive":
            pool.reset_loader_iter()
            steps_per_epoch = len(train_loader)
            iterable = range(steps_per_epoch)
        elif strategy == "block":
            iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        for itr in pbar:
            # update current K if using k schedule
            if strategy == "progressive" and next_k_idx < len(k_schedule) and global_step == k_schedule[next_k_idx][1]:
                current_k = k_schedule[next_k_idx][0]
                if is_main:
                    print(f"[K-SWITCH] Step {global_step}: K={current_k}")

                pool = make_pool(current_k)
                pool.reset_loader_iter()
                next_k_idx += 1

            optimizer.zero_grad()
            # to enable flashattention, we do the autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy == "progressive":
                    loss = 0.0
                    for n in range(num_block):
                        xt_e, x0_e, pm_e = pool.get_block_batch(n)
                        logits = model(xt_e)
                        log_probs = F.log_softmax(logits, dim=-1)

                        # loss
                        loss_blk = mdm_loss_fn(log_probs, x0_e, xt_e, mask_id, prompt_mask = pm_e)
                        loss_blk.backward()
                        loss += loss_blk.item()

                        # progressive unmasking update
                        pool.update_block(n, log_probs)

                    pool.finish_step()

                elif strategy == "block":
                    batch = itr
                    input_ids = batch["labels"].to(device)
                    prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None

                    # get the block size
                    block_size = train_cfg.block_size
                    loss = mdm_loss_block(model, input_ids, block_size, mask_id, prompt_mask = prompt_mask)
                else:
                    raise ValueError(f"Invalid training strategy: {strategy}")
            
            # loss.backward()
            if train_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            if train_cfg.ema is not None:
                ema.update(ema_params)
            scheduler.step()
            global_step += 1
            
            if is_main:
                pbar.set_postfix(loss=loss, lr=optimizer.param_groups[0]["lr"])

                if global_step % train_cfg.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss {loss}")
                    if cfg.wandb.wandb:
                        wandb.log({"loss": loss}, step=global_step)

                        gn = grad_norm(model.parameters())
                        wandb.log({"grad_norm": gn}, step=global_step)

                        if strategy == "progressive":
                            wandb.log({"current_k": current_k}, step=global_step)

            if global_step % train_cfg.eval_steps == 0:
                model.eval()

                # validaton on the downstream task; disabled when we use EMA
                if train_cfg.ema is None:
                    val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                else:
                    val_acc_dict = None

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())
                   
                    with torch.inference_mode():
                        # validaton on the downstream task
                        val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                    ema.restore(model_to_ema.parameters())
                
                if is_main:
                    # eval acc logging
                    for key, value in val_acc_dict.items():
                        print(f"Epoch {epoch+1}, Step {global_step}, Validation Accuracy {key}: {value}")
                        if cfg.wandb.wandb:
                            if train_cfg.ema is not None:
                                wandb.log({"ema_val_acc_" + key: value}, step=global_step)
                            else:
                                wandb.log({"val_acc_" + key: value}, step=global_step)
                    
                    if global_step % train_cfg.save_steps == 0 and train_cfg.ema is not None:
                        saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, global_step, val_loss = None, extra = val_acc_dict)
                        if saved_path is not None:
                            print(f"EMA Model saved to: {saved_path}")

                model.train()
    
    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg
    cfg = OmegaConf.load(cfg_path)
    main(cfg)
