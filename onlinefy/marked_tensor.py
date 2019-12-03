import torch
import copy 
import functools
import random

class TemporalStruct(object):
    """Right now we do not support breaking down the temporal dimension into multiple ones. 
    Probably support that in the future?"""
    def __init__(self, marked_dim):
        self.marked_dim = marked_dim
        self.sub_dim_prev = 1
        self.sub_dim_next = 1
        self.offset = 0

    def copy(self):
        return copy.deepcopy(self)

    def convert(self, shape_old, shape_new):
        def prod(array):
            prod_ = 1
            for ele in array:
                prod_ *= ele
            return prod_
        assert prod(shape_old) == prod(shape_new), "Num of elements does not match"
        dims_front = prod(shape_old[:self.marked_dim])
        new_marked_dim = self._locate_dim(shape_new, dims_front * self.sub_dim_prev)
        dim_t = shape_old[self.marked_dim] // self.sub_dim_prev // self.sub_dim_next
        new_dim_with_t = shape_new[new_marked_dim]
        new_dims_front = prod(shape_new[:new_marked_dim])

        assert dims_front * self.sub_dim_prev % new_dims_front == 0
        self.sub_dim_prev = dims_front * self.sub_dim_prev // new_dims_front
        assert new_dim_with_t % dim_t == 0
        assert new_dim_with_t // dim_t % self.sub_dim_prev == 0
        self.sub_dim_next = new_dim_with_t // dim_t // self.sub_dim_prev
        self.marked_dim = new_marked_dim

    @property
    def dirty(self):
        return self.sub_dim_prev * self.sub_dim_next != 1

    @staticmethod
    def _locate_dim(array, prod):
        idx = 0
        while prod >= array[idx]:
            prod = prod // array[idx]
            idx += 1
        return idx

    def _desc(self, dim_t=None):
        desc = 'Dim={dim} '.format(dim=self.marked_dim)
        if self.dirty:
            desc += 'SubDims=[{prev} <{subdim}> {next}] '.format(prev=self.sub_dim_prev, sub_dim=dim_t, next=self.sub_dim_next)
        if self.offset > 0:
            desc += 'Offset={offset}'.format(offset=self.offset)
        return desc

    def __repr__(self):
        return 'TemporalStruct({desc})'.format(desc=self._desc())

class MarkedTensor(torch.Tensor):
    def __new__(cls, x, marked_dim, *args, **kwargs): 
        r"""nn.Parameter style does not work - it converts tensor to leaf node"""
        return super().__new__(cls, x, *args, **kwargs) 
        # return torch.Tensor._make_subclass(cls, x, x.requires_grad) 
         
    def __init__(self, x, marked_dim): 
        self.tstruct = TemporalStruct(marked_dim=marked_dim)

    @property
    def marked_dim(self):
        return self.tstruct.marked_dim        

    @property
    def dirty(self):
        return self.tstruct.dirty

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

'''
    def clone(self, *args, **kwargs): 
        new_obj = MarkedTensor(super().clone(*args, **kwargs), self.marked_dim)
        new_obj.require_grad_(self.require_grad)

    def to(self, *args, **kwargs):
        new_obj = MarkedTensor([], self.marked_dim)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj

    def contiguous(self):
        new_obj = MarkedTensor([], self.marked_dim)
        new_obj.data = super().contiguous()
        return new_obj
'''
        

def mark_tensor(tensor, temporal_struct):
    marked_tensor = MarkedTensor(tensor, 0)
    marked_tensor.tstruct = temporal_struct.copy()
    return marked_tensor
