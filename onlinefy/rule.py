import copy
import torch
from collections import deque, OrderedDict

from .marked_tensor import MarkedTensor
from .arg_parser import get_marked_tensors, construct_input
from .torch_signature import get_func_signature

def make_identical_func(func, sig, func_args):
    marked_tensors, marked_keys = get_marked_tensors(func_args)
    func_args = copy.copy(func_args)
    def identical_func(input_tensors, state):
        assert len(marked_tensors) == len(input_tensors), "Number of input tensors does not match"
        for key, input_tensor in zip(marked_keys, input_tensors):
            func_args[key] = input_tensor
        if sig is None:
            kwargs = func_args
        else:
            args, kwargs = construct_input(func_args, sig)
        
        return func(*args, **kwargs), None
    return identical_func

class BaseRule(object):
    def __init__(self, func):
        self.func = func
        self.signature = get_func_signature(func.__name__)

    def onlinefy(self, marked_tensors, func_args, results):
        return (make_identical_func(self.func, self.signature, func_args),
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
                inputs = torch.cat(state + (input_tensor, ), dim=target_dim)
                padding[tdim] = 0
                func_args['padding'] = padding
                func_args['input']  = inputs
                output = self.func(**func_args)
                new_state = state[1:] + (input_tensor, )
                return output, new_state

            initial_state_shape = list(marked_tensor.shape)
            initial_state_shape[target_dim] = 1
            initial_state = (torch.zeros(initial_state_shape, dtype=marked_tensor.dtype, device=marked_tensor.device), ) * padding[tdim]

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
            # output_tensor = torch.tensor_original.view(input_tensor, new_shape)
            output_tensor = input_tensor.view(new_shape)
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
            # return torch.tensor_original.permute(input_tensor, permute_order), None
            return input_tensor.permute(permute_order), None
        return online_permute, None, tstruct_new

class GetItemRule(BaseRule):
    '''
    Support two temporal operations: offset and step
    '''
    def __init__(self, func):
        self.func = func
        self.signature = OrderedDict(self=(None, True, False), index=(None, True, False))

    def _input_analyze(self, func_args):
        shape = func_args['self'].shape
        index = func_args['index']
        if type(index) is slice or isinstance(index, int) or index is None or index is Ellipsis:
            index = (index, )
        if type(index) is tuple:
            index = self._resolve_ellipsis(index, shape)
            for element in index:
                assert type(element) is int or type(element) is slice, \
                    "Does not support advanced indexing"
            if len(index) < len(shape):
                index += (slice(None), ) * (len(shape) - len(index))
        else:
            raise NotImplementedError
        return index

    @staticmethod
    def _resolve_ellipsis(index, shape):
        def count(iterable, value):
            count_ = 0
            for ele in iterable:
                if ele == value:
                    count_ += 1
            return count_

        if Ellipsis in index:
            ell_idx = index.index(Ellipsis)
            pre_index = index[:ell_idx]
            post_index = index[ell_idx+1:]
            fills = len(shape) - len(index) + count(index, None)
            index = pre_index + (slice(None), ) * fills + post_index

        return index

    @staticmethod
    def _check_temporal_slice(slice_):
        if type(slice_) is not slice:
            raise Exception("Not a slice for temporal dimension")

    @staticmethod
    def _trivial_slice(slice_):
        if type(slice_) is not slice:
            raise Exception("Not a slice for temporal dimension")
        if (slice_.start is None or slice_.start == 1) and (slice_.step is None or slice_.step == 1):
            return True
        else:
            return False
        
    @staticmethod
    def _locate_dim(index, original_dim):
        dim_count = 0
        dim_subtract = 0
        for idx, element in enumerate(index):
            if dim_count == original_dim:
                break
            if element is not None:
                dim_count += 1
            if type(element) is int:
                dim_subtract += 1
        return element, idx - dim_subtract, idx

    def onlinefy(self, marked_tensors, func_args, results):
        index = self._input_analyze(func_args)
        tstruct_new = marked_tensors[0].tstruct.copy()

        temporal_slice, new_marked_dim, temporal_index = self._locate_dim(index, tstruct_new.marked_dim)
        tstruct_new.marked_dim = new_marked_dim

        self._check_temporal_slice(temporal_slice)
        if self._trivial_slice(temporal_slice):
            if temporal_slice.stop is not None and temporal_slice.stop:
                index = list(index)
                index[temporal_index] = slice(temporal_slice.start, None, temporal_slice.step)
                func_args['index'] = tuple(index)
            return (make_identical_func(self.func, self.signature, func_args),
                    None,
                    tstruct_new)

        offset = temporal_slice.start
        assert offset >= 0
        step = 1 if temporal_slice.step is None else temporal_slice.step
        def online_slice(input_tensors, state):
            state['counter'] += 1
            state['queue'].append(input_tensors[0])
            if state['counter'] == step:
                state['counter'] = 0
                output = state['queue'][0]
            else:
                output = None
            return output, state
        initial_state = {'counter': 0, 'queue': deque(maxlen=step)}
        return online_slice, initial_state, tstruct_new

class ElementwiseRule(BaseRule):
    def __init__(self, func):
        self.func = func
        qualname = func.__qualname__
        if qualname.startswith('_VariableFunctions'):
            self.signature = OrderedDict(input=(None, True, False), other=(None, True, False), alpha=(1, False, False), out=(None, False, False))

        elif qualname.startswith('_TensorBase.__'):
            self.signature = get_func_signature(func.__name__[2:-2])
        
        elif qualname.startswith('_TensorBase'):
            self.signature = get_func_signature(func.__name__)
        
        else: 
            raise Exception("Do not understand function %s"%qualname)

    def _input_analyze(self, func_args):
        if 'out' in func_args and func_args['out'] is not None:
            raise Exception("In-place elementwise operation not supported yet")
        if self.func.__qualname__.startswith('_VariableFunctions'):
            tensor1 = func_args['input']
        elif self.func.__qualname__.startswith('_TensorBase'):
            tensor1 = func_args['self']
        return tensor1

    def onlinefy(self, marked_tensors, func_args, results):
        tensor1 = self._input_analyze(func_args)
        tensor1_ndim = tensor1.ndim if isinstance(tensor1, torch.Tensor) else 0
        tensor2 = func_args['other']
        tensor2_ndim = tensor2.ndim if isinstance(tensor2, torch.Tensor) else 0
        out_dims = max(tensor1_ndim, tensor2_ndim)
        assert out_dims == results.ndim

        tstruct1 = marked_tensors[0].tstruct
        if len(marked_tensors) > 1:
            tstruct2 = marked_tensors[1].tstruct
            assert marked_tensors[0].ndim - tstruct1.marked_dim == marked_tensors[1].ndim - tstruct2.marked_dim, "Temporal dimensions do not match"
        new_marked_dim = tstruct1.marked_dim + out_dims - marked_tensors[0].ndim
        tstruct_new  = tstruct1.copy()
        tstruct_new.marked_dim = new_marked_dim

        return (make_identical_func(self.func, self.signature, func_args),
                None,
                tstruct_new)

class PadRule(BaseRule):
    def __init__(self, func):
        self.func = func
        self.signature = OrderedDict(input=(None, True, False), 
                                     pad=(None, True, False),
                                     mode=('constant', False, False),
                                     value=(0, False, False))

    def _input_analyze(self, func_args):
        pad = func_args['pad']
        if func_args['mode'] == 'constant':
            pad_by_dim = tuple((pad[2*n:2*n+2] for n in range(func_args['input'].ndim)))
        elif func_args['mode'] == 'replicate':
            pad = pad[::-1]
            pad_by_dim = ((0, 0),) * 2 + tuple((pad[2*n:2*n+2][::-1] for n in range(func_args['input'].ndim - 2)))
        else:
            raise NotImplementedError
        return pad_by_dim

    def _get_pad_arg(self, func_args, pad_by_dim):
        func_args = copy.copy(func_args)
        if func_args['mode'] == 'constant':
            pad = sum(pad_by_dim, tuple())
        elif func_args['mode'] == 'replicate':
            pad = sum(pad_by_dim[2::-1], tuple())
        else:
            raise NotImplementedError
        func_args['pad'] = pad
        return func_args

    def onlinefy(self, marked_tensors, func_args, results):
        pad_by_dim = self._input_analyze(func_args)
        marked_tensor = marked_tensors[0]
        marked_dim = marked_tensor.marked_dim
        tstruct_new = marked_tensor.tstruct.copy()

        pad_size = pad_by_dim[marked_dim][0]
        if pad_size == 0:
            if pad_by_dim[marked_dim] != (0, 0):
                pad_by_dim[marked_dim] = (0, 0)
                func_args = self._get_pad_arg(func_args, pad_by_dim)
            return (make_identical_func(self.func, self.signature, func_args),
                    None,
                    tstruct_new)
        else:
            mode = func_args['mode']
            def online_pad(input_tensors, state):
                if len(state) == 0 and mode == 'replicate':
                    state.extend([input_tensors[0]] * pad_size)
                state.append(input_tensors[0])
                return state[0], state
            initial_state = deque(maxlen=pad_size + 1)
            if mode == 'constant':
                pad_shape = list(marked_tensor.shape)
                pad_shape[marked_dim] = 1
                pad_shape = tuple(pad_shape)

                content = torch.zeros(pad_shape, dtype=marked_tensor.dtype, device=marked_tensor.device)

                initial_state.extend([content] * pad_size)
            return (online_pad, initial_state, tstruct_new)


funcrule_dict = {
    '_VariableFunctions.conv2d': ConvRule,
    '_VariableFunctions.conv3d': ConvRule,
    '_VariableFunctions.relu': UnivariateRule,
    '_TensorBase.contiguous': UnivariateRule,
    # '_VariableFunctions.sum': ReductionRule,
    '_VariableFunctions.add': ElementwiseRule,
    '_TensorBase.add': ElementwiseRule,
    '_TensorBase.__add__': ElementwiseRule,
    '_VariableFunctions.sub': ElementwiseRule,
    '_TensorBase.sub': ElementwiseRule,
    '_TensorBase.__sub__': ElementwiseRule,
    'pad': PadRule,
    '_TensorBase.__getitem__': GetItemRule,
    '_TensorBase.view': ReshapeRule,
    '_TensorBase.permute': DimPermuteRule,
}

