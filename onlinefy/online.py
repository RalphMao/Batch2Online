
from .inject import inject_torch, uninject_torch

class OnlineComprehension(object):
    def __init__(self, debug=False):
        self.session = {}
        self.session['debug'] = debug

    def __enter__(self):
        self._init_session()
        inject_torch(self.session)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        uninject_torch()

    def _init_session(self):
        self.session['graph'] = []
        self.session['states'] = {}
        self.session['temp_vars'] = {}

    def get_online_func(self, inputs, outputs):
        output_ids = [id(output) for output in outputs]
        input_ids = [id(input) for input in inputs]
        output2node_mapping = dict([(funcnode.output, funcnode) for funcnode in self.session['graph']])
        nodelist = []
        for node in self.session['graph'][::-1]:
            if node.output in outputs:
                nodelist.insert(0, node)
                outputs.remove(node.output)
                unresolved_inputs = [input for input in node.inputs if input not in inputs]
                outputs.extend(unresolved_inputs)
        assert len(outputs) == 0, "Cannot resolve all outputs"
        online_func = self.get_func_from_nodes(nodelist, input_ids, output_ids)

    @staticmethod
    def get_func_from_nodes(nodelist, inputs, outputs):
        def forward_func(input_tensors, states):
            assert len(input_tensors) == len(inputs)
            exec_vars = dict(zip(inputs, input_tensors))
            new_states = {}
            for node in nodelist:
                node.execute(exec_vars, states, new_states)
            output_tensors = [exec_vars[output] for output in outputs]
            return output_tensors, new_states
        return forward_func
    
