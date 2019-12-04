import torch
import torch.nn as nn
import onlinefy
from onlinefy.marked_tensor import MarkedTensor

class TemporalModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.tconv = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(2, 1, 1), bias=False)
        self.conv2_1 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.conv2_2 = nn.Conv2d(16, 4, kernel_size=(1, 1), padding=(0, 0), bias=False)

    @staticmethod
    def conv2d(data_5d, conv2d_module):
        old_shape = data_5d.shape
        data = data_5d.view((old_shape[0] * old_shape[1],) + old_shape[2:])
        feat = conv2d_module(data)
        feat_old_shape = feat.view((old_shape[0], old_shape[1],) + feat.shape[1:])
        return feat_old_shape

    @staticmethod
    def conv3d(data_5d, conv3d_module):
        data = data_5d.permute(0, 2, 1, 3, 4)
        feat_t = conv3d_module(data)
        feat_t = feat_t.permute(0, 2, 1, 3, 4)
        feat_t = feat_t.contiguous()
        return feat_t
        

    def forward(self, video_in):
        assert video_in.dim() == 5
        feat1 = self.conv2d(video_in, self.conv1)
        feat1 = torch.relu(feat1)
        
        feat_t = self.conv3d(feat1, self.tconv)

        output_1 = self.conv2d(feat_t, self.conv2_1)
        output_2 = self.conv2d(feat_t, self.conv2_2)
        # output_2 = torch.sum(output_2, dim=4)
        
        return output_1, output_2

def test_correctness():
    video_batch = torch.ones(2,10,3,50,50, requires_grad=True)
    tmodel = TemporalModel()
    output_1, output_2 = tmodel(video_batch)


def test_onlinefy():
    video_batch = torch.ones(2,10,3,50,50, requires_grad=True)
    marked_batch = MarkedTensor(video_batch, marked_dim=1)
    tmodel = TemporalModel()

    # Reverse mode
    '''
    inject_torch()
    output_1, output_2 = tmodel(video_batch)
    online_model = onlinefy.OnlineModel()
    online_model.analyze_outputs(output_1, output_2)
    initial_state = online_model.get_initial_state()
    online_forward = online_model.forward
    uninject_torch()
    '''

    # Forward mode
    with onlinefy.OnlineComprehension(debug=True) as om:
        outputs = tmodel(marked_batch)
        online_forward, states = om.get_online_func(
            inputs=(marked_batch,),
            outputs=outputs)
    

    outputs_1 = []
    outputs_2 = []
    for t in range(video_batch.shape[1]):
        frame = video_batch[:, t:t + 1]
        outputs_frame, states = online_forward([frame], states)
        outputs_1.append(outputs_frame[0])
        outputs_2.append(outputs_frame[1])

    outputs_new_1 = torch.cat(outputs_1, dim=1)
    outputs_new_2 = torch.cat(outputs_2, dim=1)
    import IPython;IPython.embed()

# test_correctness()
test_onlinefy()
