import torch
import types
from easydict import EasyDict
import traceback

from .rule import funcrule_dict
from .node import FuncNode
from .marked_tensor import MarkedTensor, mark_tensor
from .arg_parser import parse_func_args, get_marked_tensors, get_marked_output
from . import ops

def check_func(func):
    if type(func) is types.BuiltinFunctionType:
        return True
    if type(func) is types.FunctionType:
        return True
    MethodDescriptorType = type(str.format)
    if type(func) is MethodDescriptorType:
        return True
    SlotWrapperType = type(str.__add__)
    if type(func) is SlotWrapperType:
        return True
    return False

def get_funcrule(func):
    funcname = func.__qualname__
    return funcrule_dict[funcname](func)

def print_parent_stack(name, stack_id):
    stack = traceback.extract_stack()[stack_id]
    message = "[{name}] in {filename}:{lineno}\n    {line}\n".format(
            name=name,
            filename=stack.filename,
            lineno=stack.lineno,
            line=stack.line)
    print(message)

def inject_torch_function(func, session):
    name = func.__name__
    funcrule = get_funcrule(func)
    signature = funcrule.signature
    onlinefy_func = funcrule.onlinefy
    return inject_function(func, onlinefy_func, signature, session, name)

def inject_function(func, onlinefy_func, signature, session, name):
    def marked_func(*args, **kwargs):
        if not session['activated'] or session['nested']:
            return func(*args, **kwargs)

        func_args = parse_func_args(args, kwargs, signature)
        marked_tensors, _ = get_marked_tensors(func_args)
        
        results_unmarked = func(*args, **kwargs)
        if len(marked_tensors) == 0:
            return results_unmarked

        session['nested'] = True
        online_func, init_state, output_info = onlinefy_func(marked_tensors, func_args, results_unmarked)
        session['nested'] = False

        results, output_tensor = get_marked_output(results_unmarked, output_info)

        session['graph'].append(FuncNode(online_func, init_state, marked_tensors, output_tensor, name))
        if session['debug']:
            print_parent_stack(name, -3)
        return results
    return marked_func

def inject_torch(session):
    torch.original = EasyDict()
    torch.tensor_original = EasyDict()
    torch.nnfunc_original = EasyDict()
    ops.original = EasyDict()
    
    for name, val in torch.__dict__.items():
        if check_func(val):
            torch.original[name] = val
            if val.__qualname__ in funcrule_dict:
                setattr(torch, name, inject_torch_function(val, session))
                if session['debug']:
                    print("Inject torch.{name}".format(name=val.__qualname__))
    for name, val in torch.nn.functional.__dict__.items():
        if check_func(val):
            torch.nnfunc_original[name] = val
            if val.__qualname__ in funcrule_dict:
                setattr(torch.nn.functional, name, inject_torch_function(val, session))
                if session['debug']:
                    print("Inject torch.nn.functional.{name}".format(name=val.__qualname__))
    for name in dir(torch.Tensor):
        val = getattr(torch.Tensor, name)
        if check_func(val):
            torch.tensor_original[name] = val
            if val.__qualname__ in funcrule_dict:
                setattr(torch.Tensor, name, inject_torch_function(val, session))
                if session['debug']:
                    print("Inject torch.Tensor.{name}".format(name=val.__qualname__))

    for name in dir(ops):
        val = getattr(ops, name)
        if isinstance(val, ops.CustomOp):
            ops.original[name] = val
            setattr(ops, name, inject_function(val, val.onlinefy, val.signature, session, val.name))
            if session['debug']:
                print("Inject ops.{name}".format(name=val.name))

def uninject_torch():
    for name in torch.original:
        setattr(torch, name, torch.original[name])
    for name in torch.nnfunc_original:
        setattr(torch.nn.functional, name, torch.nnfunc_original[name])
    for name in torch.tensor_original:
        setattr(torch.Tensor, name, torch.tensor_original[name])
    for name in ops.original:
        setattr(ops, name, ops.original[name])
