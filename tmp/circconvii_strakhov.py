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
        return self.fc(input)

    def get_params(self):
        return self.fc.weight, self.fc.bias


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)
    def get_params(self):
        return self.fc.weight, self.fc.bias


if __name__ == "__main__":
    a = torch.rand(10, 3, 100)
    b = a.clone().detach()
    conv = CircConv(state_dim=3, n_adj=3)
    res = conv.forward(b, 3)
    my_conv = MyCircConv(channels_in=3, adj=3)
    weight, bias = conv.get_params()
    my_conv.set_params(weight, bias)
    my_res = my_conv.forward(a)
    if torch.all(res.eq(my_res)).item():
        print('Convolution is correct')
    else:
        print('Convolution is incorrect')
    dilconv = DilatedCircConv(state_dim=3, n_adj=3)
    dilres = dilconv.forward(a, 3)
    mydilconv = MyDilatedCircConv(channels_in=3, adj=3)
    weight, bias = dilconv.get_params()
    mydilconv.set_params(weight, bias)
    mydilres = mydilconv.forward(b)
    if torch.all(res.eq(my_res)).item():
        print('Dilated Convolution is correct')
    else:
        print('Dilated Convolution is incorrect')
