import torch
import types
from easydict import EasyDict
import traceback

from .marked_tensor import MarkedTensor, mark_tensor
from .rule import funcrule_dict
from .node import FuncNode

def get_funcrule(func):
    funcname = func.__qualname__
    return funcrule_dict[funcname](func)

def print_parent_stack(stack_id):
    stack = traceback.extract_stack()[stack_id]
    message = "{filename}:{lineno}\n    {line}\n".format(
            filename=stack.filename,
            lineno=stack.lineno,
            line=stack.line)
    print(message)

def marked_func_wrapper(func, session):
    funcrule = get_funcrule(func)
    def marked_func(*args, **kwargs):
        func_args = parse_func_args(args, kwargs, funcrule.signature)
        marked_tensors, _ = get_marked_tensors(func_args)

        if len(marked_tensors) > 0:

            results_unmarked = func(*args, **kwargs)

            online_func, init_state = funcrule.onlinefy(marked_tensors, func_args, results_unmarked)
            output_info = funcrule.analyze_output(marked_tensors, func_args, results_unmarked)

            if isinstance(results_unmarked, torch.Tensor):
                results = get_marked_tensor(results_unmarked, output_info)
                output_tensor = results
            elif type(results) is tuple:
                results = (mark_tensor(results_unmarked[0], output_info), ) + results_unmarked[1:]
                output_tensor = results[0]
            else:
                raise NotImplementedError

            session['graph'].append(FuncNode(online_func, init_state, marked_tensors, output_tensor))
            if session['debug']:
                print_parent_stack(-3)
        else:
            results = func(*args, **kwargs)

        return results
    return marked_func

def inject_torch(session):
    torch.original = EasyDict()
    torch.tensor_original = EasyDict()
    torch.nnfunc_original = EasyDict()
    for name, val in torch.__dict__.items():
        if type(val) is types.BuiltinFunctionType and val.__qualname__ in funcrule_dict:
            setattr(torch, name, marked_func_wrapper(val, session))
            torch.original[name] = val
            if session['debug']:
                print("Inject torch.{name}".format(name=val.__qualname__))
    for name, val in torch.nn.functional.__dict__.items():
        if type(val) is types.BuiltinFunctionType and val.__qualname__ in funcrule_dict:
            setattr(torch.nn.functional, name, marked_func_wrapper(val, session))
            torch.nnfunc_original[name] = val
            if session['debug']:
                print("Inject torch.nn.functional.{name}".format(name=val.__qualname__))
    for name in dir(torch.Tensor):
        val = getattr(torch.Tensor, name)
        if type(val) is type(torch.Tensor.new) and val.__qualname__ in funcrule_dict:
            setattr(torch.Tensor, name, marked_func_wrapper(val, session))
            torch.tensor_original[name] = val
            if session['debug']:
                print("Inject torch.Tensor.{name}".format(name=val.__qualname__))

def uninject_torch():
    for name in torch.original:
        setattr(torch, name, torch.original[name])
    for name in torch.nnfunc_original:
        setattr(torch.nn.functional, name, torch.nnfunc_original[name])
    for name in torch.tensor_original:
        setattr(torch.Tensor, name, torch.tensor_original[name])
