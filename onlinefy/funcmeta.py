
def make_identical_func(func, input_info):
    marked_args, args, kwargs = input_info
    def identical_func(input_tensor, state):
        for marked_idx in marked_args:
            args[marked_idx] = input_tensor
        return func(*args, **kwargs), None
    return identical_func

class BaseRule(object):
    def __init__(self, func):
        self.func = func

    def onlinefy(self, input_info):
        pass

    def analyze_output(self, input_info):
        pass

class UnivariateRule(BaseRule):
    def onlinefy(self, input_info):
        marked_args, args, kwargs = input_info
        assert len(marked_args) == 1, "Univarite function cannot take more than two Tensors as input"
        return make_identical_func(self.func, input_info)

    def analyze_output(self, input_info):
        marked_args, args, kwargs = input_info
        input_tensor = args[marked_args[0]]
        return input_tensor.tstruct

class ConvRule(BaseRule):
    def __init__(self, func):
        super().__init__(func)
        self.convdim = int(func.__qualname__[:-2])

    def onlinefy(self, input_info):
        marked_args, args, kwargs = input_info
        assert len(marked_args) == 1 and marked_args[0] == 0
        marked_tensor = args[0]
        if marked_tensor.marked_dim + self.convdim < marked_tensor.ndim
            return make_identical_func(self.func, input_info)
        else:
            tdim = marked_tensor.marked_dim - marked_tensor.ndim + self.convdim
            assert not marked_tensor.dirty 
            target_dim = marked_tensor.marked_dim
            def online_conv(input_tensor, state):
                inputs = torch.original.cat(state + (input_tensor, ), dim=target_dim)
                padding = list(kwargs['padding'])
                padding[tdim] = 0
                kwargs['padding'] = padding
                args[0] = inputs
                output = self.func(*args, **kwargs)
                new_state = state[1:] + (input_tensor, )
                return output, new_state
            return online_conv
    
    def analyze_output(self, input_info):
        marked_args, args, kwargs = input_info
        input_tensor = args[marked_args[0]]
        return input_tensor.tstruct
        

funcmeta_dict = {
    '_VariableFunctions.conv2d': ConvRule,
    '_VariableFunctions.conv3d': ConvRule,
    '_VariableFunctions.relu': UnivariateRule,
    # '_VariableFunctions.add': ElementwiseRule,
    'Tensor.view': ReshapeRule,
    'Tensor.transpose': TransposeRule,
}

