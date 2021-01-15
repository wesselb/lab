import pytest
import lab.jax as B

# noinspection PyUnresolvedReferences
from .util import check_lazy_shapes


def test_controlflowcache(check_lazy_shapes):
    cache = B.ControlFlowCache()

    assert not cache.populated
    assert str(cache) == repr(cache) == "<ControlFlowCache: populated=False>"

    # Test populating the cache.
    with cache:
        B.ones(5)
    assert str(cache) == repr(cache) == "<ControlFlowCache: populated=True>"
    assert cache.populated

    # Test that you can only get an outcome when using a cache.
    with pytest.raises(RuntimeError):
        B.control_flow.get_outcome("test")


def test_cache_cond(check_lazy_shapes):
    outcome = {}

    def f_true(x, y):
        outcome[0] = True
        return x + y

    def f_false(x, y):
        outcome[0] = False
        return 2 * (x + y)

    def f(x):
        return B.cond(x > 0, f_true, f_false, x, x)

    cache_true = B.ControlFlowCache()
    cache_false = B.ControlFlowCache()

    # Populate caches:

    with cache_true:
        assert f(1) == 2
        assert outcome[0]
        assert f(-1) == -4
        assert not outcome[0]

    with cache_false:
        assert f(-1) == -4
        assert not outcome[0]
        assert f(1) == 2
        assert outcome[0]

    # Use caches:

    with cache_true:
        assert f(-1) == -2
        assert outcome[0]
        assert f(1) == 4
        assert not outcome[0]

    with cache_false:
        assert f(1) == 4
        assert not outcome[0]
        assert f(-1) == -2
        assert outcome[0]


def test_control_flow_outcome_conversion():
    def f(x):
        B.control_flow.set_outcome("f/x", x, type=str)
        if B.control_flow.use_cache:
            return B.control_flow.get_outcome("f/x")
        else:
            return x

    control_flow_cache = B.ControlFlowCache()

    # Populate cache. The `1` will be converted to a string when it is saved.
    with control_flow_cache:
        assert f(1) == 1

    # Run with cache. Check that the conversion to a string indeed happened.
    with control_flow_cache:
        assert f(1) == "1"
