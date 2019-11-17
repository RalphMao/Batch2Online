import torch
from onlinefy.marked_tensor import MarkedTensor
from onlinefy.inject import marked_prop_wrapper

torch.conv1d = marked_prop_wrapper(torch.conv1d)
torch.sum = marked_prop_wrapper(torch.sum)
    

a = torch.ones(1,3,4, requires_grad=True)
b = MarkedTensor(a, marked_dim=1)
w = torch.ones(3,3,3, requires_grad=True)
c = torch.conv1d(b,w)
d = MarkedTensor(c, marked_dim=1)
e = torch.sum(d)
e.backward()
