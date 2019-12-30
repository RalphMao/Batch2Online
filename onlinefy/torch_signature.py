
"""
Declaration file could be found in: torch/share/ATen/Declarations.yaml
Processing is referred to as in: torch/lib/python3.6/site-packages/caffe2/contrib/aten/gen_op.py
"""
import yaml
import os
from collections import OrderedDict

def get_scalar_func(target_type):
    def scalar_func(argument_spec):
        is_variable_length = False
        if 'default' in argument_spec:
            default = target_type(argument_spec['default'])
        else:
            default = None
        return default, is_variable_length
    return scalar_func

def array_func(argument_spec):
    default = argument_spec['default'] if 'default' in argument_spec else None
    is_variable_length = 'size' not in argument_spec
    if not is_variable_length:
        default = (default, ) * argument_spec['size']
    return default, is_variable_length

def default_func(argument_spec):
    return None, False

def unimplemented_func(argument_spec):
    raise NotImplementedError

def get_full_specs():
    decl_file = os.path.join(os.path.dirname(__file__), 'resources/Declarations.yaml')
    return yaml.load(open(decl_file))

Argument_Mapping = {
    'Tensor': get_scalar_func(str),
    'Scalar': get_scalar_func(float),
    'bool': get_scalar_func(float),
    'int': get_scalar_func(int),
    'double': get_scalar_func(float),
    'int64_t': get_scalar_func(int),
    'IntArrayRef': array_func,
    'std::array<bool,2>': unimplemented_func,
    'std::array<bool,3>': unimplemented_func,
}
FullSpecs = []

def get_func_signature(funcname):
    if len(FullSpecs) == 0:
        FullSpecs.append(get_full_specs())
    fullspecs = FullSpecs[0]

    func_specs = [x for x in fullspecs if x['name'] == funcname]
    assert len(func_specs) > 0, "Cannot find spec of function %s" % funcname
    func_spec = func_specs[0]
    sig = OrderedDict()
    for argument in func_spec['arguments']:
        name = argument['name']
        data_type = argument['dynamic_type']
        process_func = Argument_Mapping.get(data_type, default_func)
        default, is_variable_length = process_func(argument)
        is_required = True
        if argument['is_nullable']:
            is_required = False
        if default is not None:
            is_required = False
        sig[name] = (default, is_required, is_variable_length)
    return sig
