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

_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None):
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
        super(Snake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, conv_type)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, adj):
        states = []

        x = self.head(x, adj)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, adj) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x

class MyCircConv(nn.Module):
    def __init__(self, channels_in, channels_out = None, adj=4):
        super(MyCircConv, self).__init__()
        self.adj = adj
        if channels_out is None:
            channels_out = channels_in
        self.convolution = nn.Conv1d(channels_in, channels_out, kernel_size=adj*2+1)

    def forward(self, x):
        x = torch.cat([x[..., -self.adj:], x, x[..., :self.adj]], dim=2)
        return self.convolution(x)

    def set_params(self, weight, bias):
        self.convolution.weight = weight
        self.convolution.bias = bias
class MyDilatedCircConv(nn.Module):
    def __init__(self, channels_in, channels_out = None, adj=4, dilation=1):
        super(MyDilatedCircConv, self).__init__()
        self.adj = adj
        self.dilation = dilation
        if channels_out is None:
            channels_out = channels_in
        self.convolution = nn.Conv1d(channels_in, channels_out, kernel_size=adj*2+1, dilation=self.dilation)

    def forward(self, x):
        x = torch.cat([x[..., -self.adj*self.dilation:], x, x[..., :self.adj*self.dilation]], dim=2)
        return self.convolution(x)

    def set_params(self, weight, bias):
        self.convolution.weight = weight
        self.convolution.bias = bias

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
    dilconv = DilatedCircConv(state_dim=3,n_adj=3)
    dilres = dilconv.forward(a,3)
    mydilconv = MyDilatedCircConv(channels_in=3,adj=3)
    weight, bias = dilconv.get_params()
    mydilconv.set_params(weight, bias)
    mydilres = mydilconv.forward(b)
    if torch.all(res.eq(my_res)).item():
        print('Dilated Convolution is correct')
    else:
        print('Dilated Convolution is incorrect')
