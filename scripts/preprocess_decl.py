import yaml
import sys
sys.path.append('.')
from onlinefy.rule import funcrule_dict
from onlinefy.torch_signature import Argument_Mapping

def truncate_unknown_arg(func_spec):
    arguments = func_spec['arguments']
    new_arg = []
    for argument in arguments:
        if argument['dynamic_type'] not in Argument_Mapping:
            if not argument['is_nullable'] and 'default' not in argument:
                raise Exception("Cannot understand required argument type:%s"%argument['dynamic_type'])
            else:
                print("Warning: unknown type %s, ignored"%argument['dynamic_type'])
        else:
            new_arg.append(argument)

    func_spec['arguments'] = new_arg
    return func_spec
    
def main():
    decl_file = 'resources/Declarations.yaml'
    decl = yaml.load(open(decl_file))
    specs = []
    for key in funcrule_dict:
        funcname = key.split('.')[-1]
        func_specs = [x for x in decl if x['name'] == funcname]
        if len(func_specs) > 0:
            # assert len(func_specs) == 1, "Func <%s> has multiple declarations"%funcname
            specs.append(truncate_unknown_arg(func_specs[0]))

    yaml.dump(specs, open('onlinefy/resources/Declarations.yaml', 'w'), default_flow_style=False)

main()
