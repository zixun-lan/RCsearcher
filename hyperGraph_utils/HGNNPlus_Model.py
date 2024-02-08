import torch
from dhg.nn import HGNNPConv


class HGNNPlusEncoder(torch.nn.Module):
    def __init__(self, layer_info, drop_rate=0.5, res=False):
        super(HGNNPlusEncoder, self).__init__()
        r"""
        layer_info (list): first elements is dimension of in_channel, the rest elemeents are dimensions of hidden layers
        [in_chanel, hidden, hidden, ...] 
        """

        self.res = res
        self.layers = torch.nn.ModuleList(
            [HGNNPConv(in_channels=layer_info[i], out_channels=layer_info[i + 1], drop_rate=drop_rate)
             for i in range(len(layer_info) - 1)])

    def forward(self, hg, X):
        for hgconv in self.layers:
            if self.res:
                tmp_x = X
                X = hgconv(X=X, hg=hg)
                X = X + tmp_x
            else:
                X = hgconv(X=X, hg=hg)
        return X



