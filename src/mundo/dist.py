"""Data parallelism."""

import os

import torch
import torch.distributed as td


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0")) if world_size() > 1 else 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def init(device: str) -> None:
    if world_size() == 1:
        return
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank())
        td.init_process_group("nccl", device_id=torch.device("cuda", local_rank()))
    else:
        td.init_process_group("gloo")


def finish() -> None:
    if td.is_initialized():
        td.destroy_process_group()


def barrier() -> None:
    if world_size() > 1:
        td.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Differentiable mean of `t` over all ranks."""
    if world_size() == 1:
        return t
    r = t.detach().clone()
    td.all_reduce(r)
    r /= world_size()
    return r + (t - t.detach()) if t.requires_grad else r


def broadcast_flag(flag: bool) -> bool:
    if world_size() == 1:
        return flag
    dev = "cuda" if td.get_backend() == "nccl" else "cpu"
    t = torch.tensor([int(flag)], device=dev)
    td.broadcast(t, 0)
    return bool(t.item())


def per_rank_batch(batch: int) -> int:
    n = world_size()
    if batch % n:
        raise ValueError(f"train.batch={batch} is not divisible by the {n} ranks")
    return batch // n
