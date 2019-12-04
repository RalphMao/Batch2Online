import yaml
from onlinefy.rule import funcrule_dict

def main():
    decl_file = 'resources/Declarations.yaml'
    decl = yaml.load(open(decl_file))
    specs = []
    for key in funcrule_dict:
        funcname = key.split('.')[-1]
        func_specs = [x for x in decl if x['name'] == funcname]
        assert len(func_specs) == 1
        specs.extend(func_specs)

    yaml.dump(specs, open('onlinefy/resources/Declarations.yaml', 'w'))

main()
