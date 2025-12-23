import os
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize distributed training environment.

    Uses environment variables set by torchrun (LOCAL_RANK, RANK, WORLD_SIZE).

    Returns:
        tuple[int, int, int]: (local_rank, global_rank, world_size).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return local_rank, global_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
