
def get_tensor_id(tensor):
    # return id(tensor)
    return (id(tensor), tensor._version)

class FuncNode(object):
    def __init__(self, func, state, inputs, output, name="default"):
        self.func = func
        self.state = state
        self.inputs = [get_tensor_id(tensor) for tensor in inputs]
        self.output = get_tensor_id(output)
        self.state_id = id(state) if state is not None else 0
        self.name = name

    def execute(self, exec_vars, states, new_states):

        func_input_tensors = [exec_vars[input] for input in self.inputs]
        state = None if self.state_id == 0 else states[self.state_id]
        output_tensor, new_state = self.func(func_input_tensors, state)
        if self.state_id != 0:
            new_states[self.state_id] = new_state
        exec_vars[self.output] = output_tensor

    def __repr__(self):
        desc = "Node <{func}> of {num} inputs".format(
                func=self.name, 
                num=len(self.inputs),
                )
        return desc
