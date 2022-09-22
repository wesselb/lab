import torch

from . import dispatch, B, Numeric
from ..random import _randcat_last_first
from ..types import TorchNumeric, TorchDType, Int, TorchRandomState
from ..util import compress_batch

__all__ = []


@dispatch
def create_random_state(_: TorchDType, seed: Int = 0):
    state = torch.Generator(device=B.ActiveDevice.active_name)
    state.manual_seed(seed)
    return state


@dispatch
def global_random_state(_: TorchDType):
    if B.ActiveDevice.active_name in {None, "cpu"}:
        return torch.random.default_generator
    else:
        parts = B.ActiveDevice.active_name.lower().split(":")

        if len(parts) == 0 or parts[0] not in {"cuda", "gpu"}:
            raise RuntimeError(f'Unknown active device "{B.ActiveDevice.active_name}".')

        # Ensure that the generators are available.
        if len(torch.cuda.default_generators) == 0:
            torch.cuda.init()

        if len(parts) == 1:
            return torch.cuda.default_generators[0]
        else:
            return torch.cuda.default_generators[int(parts[1])]


@dispatch
def set_global_random_state(state: TorchRandomState):
    global_gen = global_random_state.invoke(TorchDType)(None)
    global_gen.set_state(state.get_state())


@dispatch
def rand(state: TorchRandomState, dtype: TorchDType, *shape: Int):
    return state, torch.rand(
        shape,
        dtype=dtype,
        device=B.ActiveDevice.active_name,
        generator=state,
    )


@dispatch
def rand(dtype: TorchDType, *shape: Int):
    return rand(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def randn(state: TorchRandomState, dtype: TorchDType, *shape: Int):
    return state, torch.randn(
        shape,
        dtype=dtype,
        device=B.ActiveDevice.active_name,
        generator=state,
    )


@dispatch
def randn(dtype: TorchDType, *shape: Int):
    return randn(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def randcat(state: TorchRandomState, p: TorchNumeric, n: Int):
    p, uncompress = compress_batch(p, 1)
    inds = torch.multinomial(p, n, replacement=True, generator=state)
    inds = uncompress(inds)
    inds = _randcat_last_first(inds)
    return state, inds


@dispatch
def randcat(p: TorchNumeric, *shape: Int):
    return randcat(B.global_random_state(p), p, *shape)[1]


@dispatch
def randint(
    state: TorchRandomState,
    dtype: TorchDType,
    *shape: Int,
    lower: Int = 0,
    upper: Int,
):
    dtype = B.dtype_int(dtype)
    return state, torch.randint(
        lower,
        upper,
        shape,
        dtype=dtype,
        device=B.ActiveDevice.active_name,
        generator=state,
    )


@dispatch
def randint(dtype: TorchDType, *shape: Int, lower: Int = 0, upper):
    state = global_random_state(dtype)
    return randint(state, dtype, *shape, lower=lower, upper=upper)[1]


@dispatch
def randperm(state: TorchRandomState, dtype: TorchDType, n: Int):
    dtype = B.dtype_int(dtype)
    return state, torch.randperm(
        n,
        dtype=dtype,
        device=B.ActiveDevice.active_name,
        generator=state,
    )


@dispatch
def randperm(dtype: TorchDType, n: Int):
    return randperm(global_random_state(dtype), dtype, n)[1]


@dispatch
def randgamma(
    state: TorchRandomState,
    dtype: TorchDType,
    *shape: Int,
    alpha: Numeric,
    scale: Numeric,
):
    alpha = B.to_active_device(B.cast(dtype, alpha))
    scale = B.to_active_device(B.cast(dtype, scale))
    alpha, scale = torch.broadcast_tensors(alpha, scale)
    alpha = B.repeat(alpha, *shape)
    scale = B.repeat(scale, *shape)
    return state, torch._standard_gamma(alpha, generator=state) * scale


@dispatch
def randgamma(dtype: TorchDType, *shape: Int, alpha: Numeric, scale: Numeric):
    state = global_random_state(dtype)
    return randgamma(state, dtype, *shape, alpha=alpha, scale=scale)[1]
