import torch
import weakref
import itertools
import traceback
import time
from contextlib import contextmanager

_id_gen = itertools.count()

def track_tensor(t, label):
    tid = next(_id_gen)
    size = t.numel() * t.element_size()
    t_create = time.time()
    print(f"[CREATE] id={tid}, {label}, "
          f"dev={t.device}, shape={tuple(t.shape)}, "
          f"size={size/1024**3:.3f} GB")
    traceback.print_stack(limit=8)

    def _on_free(tid=tid, size=size, label=label, t_create=t_create):
        dt = time.time() - t_create
        print(f"[FREE]   id={tid}, {label}, size={size/1024**3:.3f} GB, "
              f"lifetime={dt:.3f}s")

    weakref.finalize(t, _on_free)
    return t


def _wrap_factory(fn, name):
    def wrapped(*args, **kwargs):
        t = fn(*args, **kwargs)
        if isinstance(t, torch.Tensor) and t.device.type == "cpu":
            track_tensor(t, f"factory:{name}")
        return t
    return wrapped


_originals = {
    "empty": torch.empty,
    "zeros": torch.zeros,
    "ones": torch.ones,
    "randn": torch.randn,
    "tensor": torch.tensor,
    "from_numpy": torch.from_numpy,
}
_tensor_to_orig = torch.Tensor.to
_tensor_cpu_orig = torch.Tensor.cpu


def _to_hook(self, *args, **kwargs):
    t = _tensor_to_orig(self, *args, **kwargs)
    if isinstance(t, torch.Tensor) and t.device.type == "cpu":
        track_tensor(t, "to(cpu)")
    return t


def _cpu_hook(self, *args, **kwargs):
    t = _tensor_cpu_orig(self, *args, **kwargs)
    if isinstance(t, torch.Tensor) and t.device.type == "cpu":
        track_tensor(t, "cpu()")
    return t


@contextmanager
def patch_torch_allocation():
    """Context manager: trace CPU tensor creation/migration within with block, restore originals on exit."""
    # apply patches
    for name, orig in _originals.items():
        setattr(torch, name, _wrap_factory(orig, name))
    torch.Tensor.to = _to_hook
    torch.Tensor.cpu = _cpu_hook
    try:
        yield
    finally:
        # restore originals
        for name, orig in _originals.items():
            setattr(torch, name, orig)
        torch.Tensor.to = _tensor_to_orig
        torch.Tensor.cpu = _tensor_cpu_orig
