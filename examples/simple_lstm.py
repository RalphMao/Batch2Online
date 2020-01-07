import torch
import torch.nn as nn
import onlinefy
from onlinefy.marked_tensor import MarkedTensor
import onlinefy.ops as ops

class TemporalModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.lstm = nn.LSTM(16*20*20, 200)

    @staticmethod
    def conv2d(data_5d, conv2d_module):
        old_shape = data_5d.shape
        data = data_5d.view((old_shape[0] * old_shape[1],) + old_shape[2:])
        feat = conv2d_module(data)
        feat_old_shape = feat.view((old_shape[0], old_shape[1],) + feat.shape[1:])
        return feat_old_shape

    def forward(self, video_in):
        assert video_in.dim() == 5
        feat1 = self.conv2d(video_in, self.conv1)
        feat1 = torch.relu(feat1)
        
        feat1_shape = feat1.shape
        feat1 = feat1.view(feat1_shape[0], feat1_shape[1], feat1_shape[2] * feat1_shape[3] * feat1_shape[4])
        # feat1 = feat1.transpose(0, 1)
        feat1 = feat1.permute(1, 0, 2)
        # feat1 = feat1.view(feat1_shape[0], feat1_shape[1], -1)
        hidden_init = (torch.zeros(1, 2, 200), torch.zeros(1, 2, 200))
        feat_t = ops.scan(feat1, hidden_init, self.lstm, keepdim=True)

        
        return feat_t

def test_correctness():
    video_batch = torch.ones(2,10,3,20,20, requires_grad=True)
    tmodel = TemporalModel()
    output_1 = tmodel(video_batch)


def test_onlinefy():
    video_batch = torch.ones(2,10,3,20,20, requires_grad=True)
    marked_batch = MarkedTensor(video_batch, marked_dim=1)
    tmodel = TemporalModel()

    with onlinefy.TemporalComprehension(debug=True) as tm:
        outputs = tmodel(marked_batch)
        online_forward, states = tm.get_online_func(
            inputs=(marked_batch,),
            outputs=(outputs,))
    

    output_list = []
    for t in range(video_batch.shape[1]):
        frame = video_batch[:, t:t + 1]
        outputs_frame, states = online_forward([frame], states)
        output_list.append(outputs_frame[0])

    outputs_new = torch.stack(output_list, dim=0)
    print(torch.std(outputs_new - outputs))

# test_correctness()
test_onlinefy()
