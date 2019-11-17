
from .global_cache import global_cache

class OnlineModel(object):
    def __init__(self):
        pass

    def __enter__(self):
        self._init_cache(global_cache)

    def __exit__(self, exception_type, exception_value, traceback):
        self._clear_cache(global_cache)

    def _init_cache(self, cache):
        if not cache['lock']:
            cache['lock'] = True
            cache['graph'] = []
            cache['states'] = []
        else:
            raise Exception("Cannot access global cache. Already occupied by others?")

    def _clear_cache(self, cache):
        assert cache['lock']
        cache['lock'] = False
