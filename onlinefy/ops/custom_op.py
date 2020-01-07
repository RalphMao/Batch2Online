
import inspect 
from collections import OrderedDict
import traceback

class CustomOp(object):
    def __init__(self, batch_func, onlinefy_func, name=None):
        self.signature = self._get_signature(batch_func)
        self.forward_batch = batch_func
        self.onlinefy = onlinefy_func
        self.name = batch_func.__name__ if name is None else name

    @staticmethod
    def _get_signature(func):
        pysig = inspect.signature(func)
        sig = OrderedDict()
        for param_name in pysig.parameters:
            default = pysig.parameters[param_name].default
            if default == inspect._empty:
                sig[param_name] = (None, True, False)
            else:
                sig[param_name] = (default, False, False)
        return sig

    def __call__(self, *args, **kwargs):
        return self.forward_batch(*args, **kwargs)
