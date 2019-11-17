import torch
import types

from .marked_tensor import MarkedTensor, get_dim_struct, get_marked_tensor
from .global_cache import global_cache
from .funcrule import get_funcrule

def input_analyze(args, kwargs):
    # Assumes MarkedTensor can only be in args
    marked_args = [idx for idx in range(len(args)) if type(args[idx]) is MarkedTensor]
    return marked_args, args, kwargs

def mark_tensors(results, output_info):
    # Not sure if it is correct. For now just do it
    if type(results) is torch.Tensor and marked_dim is not None and results.ndim > 0:
        results = MarkedTensor(results, marked_dim)

def func_analyze(func, input_info):
    marked_args, args, kwargs = input_info
    marked_tensor_input = args[marked_args[0]]
    output_info = [marked_tensor_input.tstruct]
    return output_info

def mark_tensors(output, output_info):
    output[0] = get_marked_tensor(output[0], output_info[0])

def marked_prop_wrapper(func):
    def marked_func(*args, **kwargs):
        
        input_info = input_analyze(args, kwargs)
        if len(input_info[0]) > 0:
            funcrule = get_funcrule(func)
            online_func = funcrule.onlinefy(input_info)

            results = func(*args, **kwargs)

            output_info = funcrule.analyze_output(input_info, results)

            global_cache.append(FuncNode(online_func, input_info))
        
            results = mark_tensors(results, output_info)
        else:
            results = func(*args, **kwargs)

        return results
    return marked_func

def inject_torch():
    for attr in dir(torch):
        if attr in dir(object):
            continue
        if type(attr) is not types.BuiltinFunctionType:
            continue
        setattr(torch, attr, marked_prop(getattr(torch, attr)))

def inject_tensor():
    torch.Tensor.__add__ = marked_prop(torch.Tensor.__add__)
