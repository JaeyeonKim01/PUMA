import os, json
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, load_dataset_builder
from transformers import AutoTokenizer
from torch.utils.data import Dataset, random_split


def _get_train_size_fallback():
    # If builder metadata fails, you can hardcode like 11_800_000
    return 11_800_000

def pretokenize_tinygsm(
    out_dir: str,
    tokenizer_name: str = "Qwen/Qwen2-0.5B",
    max_len: int = 512,
    sep: str = "\n",
    batch_size: int = 2048,
    streaming: bool = True,
    limit: int | None = None,
):
    """
    Tokenize TinyGSM into fixed-length sequences and save as memmaps.

    Sequence format (MDM-style, one example = one seq):
      ids_raw = prompt_ids + sep_ids + answer_ids
      if len(ids_raw) >= max_len:
         ids = ids_raw[:max_len-1] + [EOS]
      else:
         ids = ids_raw + [EOS] * (max_len - len(ids_raw))   # EOS used as padding too

    prompt_mask convention (per your request):
      - True for prompt tokens
      - False for answer tokens + pad tokens
    """
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    sep_ids = tok(sep, add_special_tokens=False).input_ids
    sep_len = len(sep_ids)

    # Determine dataset size (so we can allocate memmaps)
    if not streaming:
        ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=False)
        N = len(ds)
    else:
        try:
            builder = load_dataset_builder("TinyGSM/TinyGSM")
            N = builder.info.splits["train"].num_examples
        except Exception:
            N = _get_train_size_fallback()

        ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)

    if limit is not None:
        N = min(N, int(limit))

    if max_len % 8 != 0:
        raise ValueError("max_len must be divisible by 8 to pack prompt_mask with packbits cleanly.")

    mask_bytes = max_len // 8  # 512 -> 64

    labels_path = os.path.join(out_dir, "labels.bin")
    mask_path   = os.path.join(out_dir, "prompt_mask.bin")
    meta_path   = os.path.join(out_dir, "meta.json")

    # Allocate memmaps
    labels_mm = np.memmap(labels_path, mode="w+", dtype=np.uint32, shape=(N, max_len))
    mask_mm   = np.memmap(mask_path,   mode="w+", dtype=np.uint8,  shape=(N, mask_bytes))

    def batched(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == n:
                yield buf
                buf = []
        if buf:
            yield buf

    def pack_mask(mask_bool_1d: np.ndarray) -> np.ndarray:
        # Use bitorder='little' for consistency on unpack
        return np.packbits(mask_bool_1d.astype(np.uint8), axis=-1, bitorder="little")

    written = 0
    pbar_total = N
    pbar = tqdm(total=pbar_total, desc=f"Pretokenizing TinyGSM -> {out_dir}")

    for batch in batched(ds, batch_size):
        if written >= N:
            break
        if written + len(batch) > N:
            batch = batch[: (N - written)]

        prompts = [(ex.get("question") or "").strip() for ex in batch]
        answers = [(ex.get("code") or "").strip() for ex in batch]

        # Batched tokenize
        p_ids_batch = tok(prompts, add_special_tokens=False).input_ids
        a_ids_batch = tok(answers, add_special_tokens=False).input_ids

        for p_ids, a_ids in zip(tqdm(p_ids_batch, total=len(p_ids_batch), desc="Tokenizing TinyGSM (streaming)"), tqdm(a_ids_batch, total=len(a_ids_batch), desc="Tokenizing TinyGSM (streaming)")):
            # Build raw ids
            raw_ids = p_ids + sep_ids + a_ids
            prompt_len_raw = len(p_ids) + sep_len  # boundary in raw_ids
            pm = np.zeros((max_len, ) , dtype=np.bool_) # set False

            if len(raw_ids) >= max_len:
                # truncate + append EOS
                ids = raw_ids[: max_len - 1] + [eos_id]
                prompt_boundary = min(prompt_len_raw, max_len - 1)
                if prompt_boundary > 0:
                    pm[:prompt_boundary] = True
            else:
                ids = raw_ids + [eos_id] * (max_len - len(raw_ids))
                prompt_boundary = min(prompt_len_raw, max_len)
                if prompt_boundary > 0:
                    pm[:prompt_boundary] = True

            # Write to memmaps
            labels_mm[written, :] = np.asarray(ids, dtype=np.uint32)
            mask_mm[written, :]   = pack_mask(pm)
            written += 1
            pbar.update(1)

            if written >= N:
                break

    pbar.close()

    # Flush to disk
    labels_mm.flush()
    mask_mm.flush()

    meta = {
        "dataset": "TinyGSM/TinyGSM",
        "split": "train",
        "tokenizer": tokenizer_name,
        "max_len": max_len,
        "sep": sep,
        "eos_id": int(eos_id),
        "num_examples": int(written),
        "labels_dtype": "uint32",
        "prompt_mask_packed": True,
        "prompt_mask_bitorder": "little",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote {written:,} examples")
    print(f"- {labels_path} (uint32) shape=({written},{max_len})")
    print(f"- {mask_path}   (packed uint8) shape=({written},{max_len//8})")
    print(f"- {meta_path}")



class TinyGSMDataset(Dataset):
    """
    Loads pretokenized TinyGSM from:
      - labels.bin       uint32 [N, max_len]
      - prompt_mask.bin  uint8  [N, max_len/8] (packed bits)
      - meta.json
    Returns:
      {"labels": LongTensor[max_len], "prompt_mask": BoolTensor[max_len]}
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.max_len = int(self.meta["max_len"])
        self.N = int(self.meta["num_examples"])
        self.bitorder = self.meta.get("prompt_mask_bitorder", "little")

        labels_path = os.path.join(data_dir, "labels.bin")
        mask_path   = os.path.join(data_dir, "prompt_mask.bin")

        self.labels_mm = np.memmap(labels_path, mode="r", dtype=np.uint32, shape=(self.N, self.max_len))
        self.mask_mm   = np.memmap(mask_path,   mode="r", dtype=np.uint8,  shape=(self.N, self.max_len // 8))

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        labels = torch.from_numpy(self.labels_mm[idx].astype(np.int64))  # to torch long
        packed = self.mask_mm[idx]
        mask = np.unpackbits(packed, bitorder=self.bitorder)[: self.max_len].astype(np.bool_)
        prompt_mask = torch.from_numpy(mask)

        return {"labels": labels, "prompt_mask": prompt_mask}

def split_tinygsm(data_dir: str, val_ratio: float = 0.05, seed: int = 2025):
    dataset = TinyGSMDataset(data_dir)
    n = len(dataset)

    n_val = int(n * val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(dataset, [n_train, n_val], generator=g)
    return train_data, val_data



if __name__ == "__main__":
    out_dir = "data/tiny_gsm"
    pretokenize_tinygsm(out_dir=out_dir)