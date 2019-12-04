import copy
import torch
from .marked_tensor import MarkedTensor
from .arg_parser import get_marked_tensors, construct_input
from .torch_signature import get_func_signature

def make_identical_func(func, func_args):
    marked_tensors, marked_keys = get_marked_tensors(func_args)
    func_args = copy.copy(func_args)
    def identical_func(input_tensors, state):
        assert len(marked_tensors) == len(input_tensors), "Number of input tensors does not match"
        for key, input_tensor in zip(marked_keys, input_tensors):
            func_args[key] = input_tensor
        args, kwargs = construct_input(func_args)
        
        return func(*args, **kwargs), None
    return identical_func

class BaseRule(object):
    def __init__(self, func):
        self.func = func
        self.signature = get_func_signature(func.__name__)

    def onlinefy(self, marked_tensors, func_args, results):
        return (make_identical_func(self.func, func_args),
               None, 
               marked_tensors[0].tstruct.copy())

class UnivariateRule(BaseRule):
    pass

class ConvRule(BaseRule):
    def __init__(self, func):
        super().__init__(func)
        self.convdim = int(func.__qualname__[-2])

    def onlinefy(self, marked_tensors, func_args, results):
        marked_tensor = func_args["input"]
        weight = func_args["weight"]
        assert len(marked_tensors) == 1 and marked_tensors[0] is marked_tensor
        if marked_tensor.marked_dim + self.convdim < marked_tensor.dim():
            return super().onlinefy(marked_tensors, func_args, results)
        else:
            tdim = marked_tensor.marked_dim - marked_tensor.dim() + self.convdim
            assert not marked_tensor.dirty 
            target_dim = marked_tensor.marked_dim
            kernel_size = weight.shape[target_dim]
            assert func_args['stride'][tdim] == 1
            assert func_args['dilation'][tdim] == 1
            assert func_args['padding'][tdim] == kernel_size - 1
            padding = list(func_args['padding'])
            def online_conv(input_tensors, state):
                input_tensor = input_tensors[0]
                inputs = torch.original.cat(state + (input_tensor, ), dim=target_dim)
                padding[tdim] = 0
                func_args['padding'] = padding
                func_args['input']  = inputs
                output = self.func(**func_args)
                new_state = state[1:] + (input_tensor, )
                return output, new_state

            initial_state_shape = list(marked_tensor.shape)
            initial_state_shape[target_dim] = 1
            initial_state = (torch.zeros(initial_state_shape), ) * padding[tdim]

            t_struct_new = marked_tensor.tstruct.copy()
            t_struct_new.offset += padding[tdim] + 1 - kernel_size 

            return online_conv, initial_state, t_struct_new

class ReshapeRule(BaseRule):
    def _input_analyze(self, func_args):
        if self.func.__name__ == "view":
            shape = func_args['size']
        else:
            raise Exception("Unknown function:%s", self.func.__qualname__)
        return shape

    def onlinefy(self, marked_tensors, func_args, results):
        marked_tensor = marked_tensors[0]
        shape = marked_tensor.shape
        out_shape = results.shape
        tstruct_new = marked_tensor.tstruct.copy()
        tstruct_new.convert(shape, out_shape)
        
        marked_dim = marked_tensor.marked_dim
        tdim = marked_tensor.tdim_size
        def online_reshape(input_tensors, state):
            input_tensor = input_tensors[0]
            new_shape = list(out_shape)
            assert new_shape[tstruct_new.marked_dim] % tdim == 0
            new_shape[tstruct_new.marked_dim] = new_shape[tstruct_new.marked_dim] // tdim
            new_shape = tuple(new_shape)
            output_tensor = torch.tensor_original.view(input_tensor, new_shape)
            return output_tensor, None
        return online_reshape, None, tstruct_new

class DimPermuteRule(BaseRule):
    def _input_analyze(self, func_args):
        if self.func.__name__ == "permute":
            permute_order = func_args['dims']
        else:
            raise Exception("Unknown function:%s", self.func.__qualname__)
        return permute_order

    def onlinefy(self, marked_tensors, func_args, results):
        permute_order = self._input_analyze(func_args)
        tstruct_new = marked_tensors[0].tstruct.copy()
        tstruct_new.marked_dim = permute_order.index(tstruct_new.marked_dim)

        def online_permute(input_tensors, state):
            input_tensor = input_tensors[0]
            return torch.tensor_original.permute(input_tensor, permute_order), None
        return online_permute, None, tstruct_new


funcrule_dict = {
    '_VariableFunctions.conv2d': ConvRule,
    '_VariableFunctions.conv3d': ConvRule,
    '_VariableFunctions.relu': UnivariateRule,
    # '_VariableFunctions.sum': ReductionRule,
    # '_VariableFunctions.add': ElementwiseRule,
    '_TensorBase.view': ReshapeRule,
    '_TensorBase.contiguous': UnivariateRule,
    '_TensorBase.permute': DimPermuteRule,
}

