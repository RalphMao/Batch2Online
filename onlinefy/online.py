
from .inject import inject_torch, uninject_torch
from .node import get_tensor_id

class TemporalComprehension(object):
    session = {
        'debug': False,
        'graph': [],
        'activated': False,
        'nested': False
        }
    def __init__(self, debug=False):
        self.session['debug'] = debug

    def __enter__(self):
        self._init_session()
        inject_torch(self.session)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        uninject_torch()
        self._clear_session()

    def _init_session(self):
        assert not self.session['activated'], "Already under temporal comprehension"
        self.session['activated'] = True
        self.session['graph'] = []

    def _clear_session(self):
        self.session['activated'] = False

    def get_online_func(self, inputs, outputs):
        output_ids = [get_tensor_id(output) for output in outputs]
        input_ids = [get_tensor_id(input) for input in inputs]
        output2node_mapping = dict([(funcnode.output, funcnode) for funcnode in self.session['graph']])
        nodelist = []
        running_output_ids = [id_ for id_ in output_ids]
        for node in self.session['graph'][::-1]:
            if node.output in running_output_ids:
                nodelist.insert(0, node)
                running_output_ids.remove(node.output)
                unresolved_inputs = [input for input in node.inputs if input not in input_ids]
                running_output_ids.extend(unresolved_inputs)
                running_output_ids = list(set(running_output_ids))
                
            else:
                print("Warning - cannot link output of func %s"%node.func.__qualname__)
        assert len(running_output_ids) == 0, "Cannot resolve all outputs"
        online_func = self.get_func_from_nodes(nodelist, input_ids, output_ids)
        states = self.get_states_from_nodes(nodelist)
        return online_func, states

    def get_func_from_nodes(self, nodelist, inputs, outputs):
        def forward_func(input_tensors, states):
            if self.session['activated']:
                print("Warning - call online function inside TC environment")
            assert len(input_tensors) == len(inputs)
            exec_vars = dict(zip(inputs, input_tensors))
            new_states = {}
            for node in nodelist:
                node.execute(exec_vars, states, new_states)
            output_tensors = [exec_vars[output] for output in outputs]
            return output_tensors, new_states
        return forward_func
    
    @staticmethod
    def get_states_from_nodes(nodelist):
        states = {}
        for node in nodelist:
            if node.state_id != 0:
                states[node.state_id] = node.state
        return states
