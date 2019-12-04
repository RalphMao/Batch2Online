'''
Mimic the behavior of torch/csrc/utils/python_arg_parser.h
Signature is an ordered dict with fields: default, is_required, is_variable_length.
Example:
sig = OrderedDict([
                  ['input', (None, True, False)],
                  ['size', (None, True, True)],
                  ])
Input is represented with ordered dict
'''

from collections import OrderedDict

from .marked_tensor import MarkedTensor

def default_arg_from_sig(sig):
    func_args = OrderedDict()
    for key, (default, is_required, is_variable_length) in sig.items():
        if is_variable_length:
            func_args[key] = []
        else:
            func_args[key] = default
    return func_args

def _isiterable(element):
    if type(element) is tuple or type(element) is list:
        return True
    else:
        return False

def parse_func_args(args, kwargs, sig):
    # This function does not perform any sanity check!
    func_args = default_arg_from_sig(sig)
    args = list(args)
    sig_idx = 0
    sig_keys = list(sig.keys())
    while len(args) > 0:
        arg = args.pop(0)
        key = sig_keys[sig_idx]
        is_variable_length = sig[key][2]
        if is_variable_length and not _isiterable(arg):
            func_args[key].append(arg)
        else:
            func_args[key] = arg
            sig_idx += 1
    for key, val in kwargs:
        func_args[key] = val
    
    return func_args

def construct_input(func_args):
    args = func_args.values()
    kwargs = {}
    return args, kwargs

def get_marked_tensors(func_args):
    keys = []
    marked_tensors = []
    for key, val in func_args.items():
        if type(val) is MarkedTensor:
            keys.append(key)
            marked_tensors.append(val)
    return marked_tensors, keys
