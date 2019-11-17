import torch
import copy 

class TemporalStruct(object):
    def __init__(self, marked_dim=0):
        self.marked_dim = 0
        self.mixin = False
        self.sub_shape = (x.shape[marked_dim],)
        self.sub_dims = (0,)
        self.offset = 0

class MarkedTensor(torch.Tensor):
    def __new__(cls, x, marked_dim, *args, **kwargs): 
        r"""nn.Parameter style does not work - it converts tensor to leaf node"""
        return super().__new__(cls, x, *args, **kwargs) 
        # return torch.Tensor._make_subclass(cls, x, x.requires_grad) 
         
    def __init__(self, x, marked_dim): 
        self.tstruct = TemporalStruct(marked_dim=marked_dim)

        self.ancestors = []
        self.meta = None

    def marked_dim(self):
        return self.tstruct.marked_dim        

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def clone(self, *args, **kwargs): 
        new_obj = MarkedTensor(super().clone(*args, **kwargs), self.marked_dim)
        new_obj.require_grad_(self.require_grad)

    def to(self, *args, **kwargs):
        new_obj = MarkedTensor([], self.marked_dim)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj

def get_marked_tensor(tensor, temporal_struct):
    marked_tensor = MarkedTensor(tensor, 0)
    marked_tensor.tstruct = copy.deepcopy(temporal_struct)
    return marked_tensor
