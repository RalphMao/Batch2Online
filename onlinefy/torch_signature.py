
"""
Declaration file could be found in: torch/share/ATen/Declarations.yaml
Processing is referred to as in: torch/lib/python3.6/site-packages/caffe2/contrib/aten/gen_op.py
"""
import yaml
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

def unimplemented_func(argument_spec):
    raise NotImplementedError

def get_full_specs():
    decl_file = os.path.join(os.path.dirname(__file__), 'resources/Declarations.yaml')
    return yaml.load(open(decl_file))

Argument_Mapping = {
    'Scalar': get_scalar_func(float),
    'bool': get_scalar_func(float),
    'int': get_scalar_func(int),
    'double': get_scalar_func(float),
    'int64_t': get_scalar_func(int),
    'IntArrayRef': array_func,
    'std::array<bool,2>': unimplemented_func,
    'std::array<bool,3>': unimplemented_func,
}
FullSpecs = get_full_specs()

def get_func_signature(funcname):
    func_specs = [x for x in FullSpecs if x['name'] == funcname]
    assert len(func_specs) > 0
    func_spec = func_specs[0]
    sig = OrderedDict()
    for argument in func_spec['arguments']:
        name = argument['name']
        data_type = argument['dynamic_type']
        process_func = Argument_Mapping[data_type]
        default, is_variable_length = process_func(argument)
        is_required = True
        if argument['is_nullable']:
            is_required = False
        if default is not None:
            is_required = False
        sig[name] = (default, is_required, is_variable_length)
    return sig
