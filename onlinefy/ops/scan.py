
import torch
from .custom_op import CustomOp
from ..marked_tensor import MarkedTensor

def scan_forward(tensor_input, init_state, update_func, dim=0, keepdim=False):
    tlen = tensor_input.shape[dim]
    output_list = []
    state_list = []
    updated_state = init_state
    for tidx in range(tlen):
        tensor_t = tensor_input.select(dim, tidx)
        if keepdim:
            tensor_t = tensor_t.unsqueeze(dim)
        output, updated_state = update_func(tensor_t, updated_state)
        output_list.append(output)
        state_list.append(updated_state)
    tensor_output = torch.stack(output_list, dim=dim)
    # tensor_state = torch.stack(state_list, dim=dim)
    return tensor_output
        
def scan_onlinefy(marked_tensors, func_args, results):
    update_func = func_args['update_func']
    assert len(marked_tensors) == 1
    assert isinstance(func_args['tensor_input'], MarkedTensor)
    tensor_input = func_args['tensor_input']
    scan_dim = func_args['dim']
    init_state = func_args['init_state']
    tstruct_new = tensor_input.tstruct.copy()
    if tensor_input.marked_dim == scan_dim:
        def online_scan(input_tensors, state):
            input_tensor = input_tensors[0]
            return update_func(input_tensor, state)
        return online_scan, init_state, tstruct_new
    else:
        return (make_identical_func(scan_forward, None, func_args),
                None,
                tstruct_new)

scan = CustomOp(scan_forward, scan_onlinefy, name='scan')
