__all__ = ["control_flow", "ControlFlowCache"]


class ControlFlow:
    """Control flow.

    Attributes:
        caching (bool): Are we currently caching?
        use_cache (bool): Are we currently using a cache?
    """

    def __init__(self):
        self._cache = None
        self._counter = -1
        self.caching = False
        self.use_cache = False

    def start_caching(self, cache):
        """Start caching.

        Args:
            cache (:class:`.control_flow.ControlFlowCache`): Cache to populate.
        """
        self._cache = cache
        self._counter = -1
        self.caching = True

    def stop_caching(self):
        """Stop caching."""
        self.caching = False

    def start_using_cache(self, cache):
        """Start using a cache.

        Args:
            cache (:class:`.control_flow.ControlFlowCache`): Cache to use.
        """
        self._cache = cache
        self._counter = -1
        self.use_cache = True

    def stop_using_cache(self):
        """Stop using a cache."""
        self.use_cache = False

    def get_outcome(self, name):
        """Get an outcome.

        Args:
            name (str): Name of the operation.
        """
        if self.use_cache:
            self._counter += 1
            return self._cache.outcomes[name, self._counter]
        else:
            raise RuntimeError("Can only get an outcome when a cache is used.")

    def set_outcome(self, name, outcome, type=None):
        """Set an outcome.

        Args:
            name (str): Name of the operation.
            outcome (object): Outcome.
            type (type, optional): Type to convert the outcome to.
        """
        if self.caching:
            self._counter += 1
            if type:
                outcome = type(outcome)
            self._cache.outcomes[name, self._counter] = outcome


control_flow = ControlFlow()


class ControlFlowCache:
    """A control flow cache.

    Attributes:
        populated (bool): Is the cache already populated?
        outcomes (dict): Outcomes.
    """

    def __init__(self):
        self.populated = False
        self.outcomes = {}

    def __enter__(self):
        if self.populated:
            control_flow.start_using_cache(self)
        else:
            control_flow.start_caching(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.populated:
            control_flow.stop_using_cache()
        else:
            self.populated = True
            control_flow.stop_caching()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"<ControlFlowCache: populated={self.populated}>"
