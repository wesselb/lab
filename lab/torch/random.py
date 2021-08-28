import torch

from . import B, dispatch
from ..types import TorchNumeric, TorchDType, Int, TorchRandomState

__all__ = []


@dispatch
def create_random_state(_: TorchDType, seed: Int = 0):
    state = torch.Generator()
    state.manual_seed(seed)
    return state


@dispatch
def global_random_state(_: TorchDType):
    return torch.random.default_generator


@dispatch
def set_global_random_state(state: TorchRandomState):
    torch.random.default_generator.set_state(state.get_state())


@dispatch
def rand(state: TorchRandomState, dtype: TorchDType, *shape: Int):
    return state, torch.rand(
        shape, dtype=dtype, device=B.ActiveDevice.active_name, generator=state
    )


@dispatch
def rand(dtype: TorchDType, *shape: Int):
    return rand(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def randn(state: TorchRandomState, dtype: TorchDType, *shape: Int):
    return state, torch.randn(
        shape, dtype=dtype, device=B.ActiveDevice.active_name, generator=state
    )


@dispatch
def randn(dtype: TorchDType, *shape: Int):
    return randn(global_random_state(dtype), dtype, *shape)[1]


@dispatch
def choice(state: TorchRandomState, a: TorchNumeric, n: Int):
    choices = a[torch.randint(a.shape[0], (n,), generator=state)]
    return state, choices[0] if n == 1 else choices


@dispatch
def choice(a: TorchNumeric, n: Int):
    return choice(global_random_state(a), a, n)[1]
