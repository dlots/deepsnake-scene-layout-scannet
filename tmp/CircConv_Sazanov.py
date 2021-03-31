import torch.nn as nn
import torch


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input, adj):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return input, self.fc(input)

    def get_weight_bias(self):
        return self.fc.weight, self.fc.bias


class MyCircConv(nn.Module):
    def __init__(self, channels, adjacent):
        super(MyCircConv, self).__init__()
        self.adjacent = adjacent
        self.convolution = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=adjacent*2+1
        )

    def forward(self, x):
        x = torch.cat([x[..., -self.adjacent:], x, x[..., :self.adjacent]], dim=2)
        return x, self.convolution(x)

    def set_weight_bias(self, weight, bias):
        self.convolution.weight = weight
        self.convolution.bias = bias


if __name__ == "__main__":
    a = torch.rand(10, 3, 1000)
    b = a.clone().detach()
    #c = a.clone().detach()

    #dil = DilatedCircConv(3, n_adj=2, dilation=1)
    #dil_res = dil.forward(c, 2)

    #print(b.dim())
    conv = CircConv(state_dim=3, n_adj=2)
    b, res = conv.forward(b, 2)

    #print(a.dim())
    my_conv = MyCircConv(channels=3, adjacent=2)
    weight, bias = conv.get_weight_bias()
    my_conv.set_weight_bias(weight, bias)
    a, my_res = my_conv.forward(a)

    if torch.all(a.eq(b)).item():
        print('tensors are equal')
    else:
        print('tensors are not equal')

