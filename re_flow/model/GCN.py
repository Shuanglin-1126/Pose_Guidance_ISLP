import torch
import torch.nn as nn
import copy as cp
import math
from typing import Optional, Union, Dict, List

from .Graph import Graph

EPS = 1e-4


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_gcn(nn.Module):
    """The basic unit of graph convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        adaptive (str): The strategy for adapting the weights of the
            adjacency matrix. Defaults to ``'importance'``.
        conv_pos (str): The position of the 1x1 2D conv.
            Defaults to ``'pre'``.
        with_res (bool): Whether to use residual connection.
            Defaults to False.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        act (str): The name of activation layer. Defaults to ``'Relu'``.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 adaptive: str = 'init',
                 conv_pos: str = 'pre',
                 with_res: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.bn0 = nn.BatchNorm1d(out_channels * self.num_subsets)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv1d(in_channels, out_channels * self.num_subsets, 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv1d(self.num_subsets * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1),
                    nn.BatchNorm1d(out_channels))
            else:
                self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        n, c, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({
                'offset': self.A + self.PA,
                'importance': self.A * self.PA
            })
        A = A_switch[self.adaptive] # (3, v, v) 3 denotes the self_link, inward, outward

        if self.conv_pos == 'pre':
            x = self.bn0(self.conv(x))
            x = x.view(n, self.num_subsets, -1, v)
            x = torch.einsum('nkcv,kvw->ncw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('ncv,kvw->nkcw', (x, A)).contiguous()
            x = x.view(n, -1, v)
            x = self.conv(x)

        return self.relu(self.bn(x) + res)


class STGCNBlock(nn.Module):
    """The basic block of STGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        residual (bool): Whether to use residual connection. Defaults to True.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 residual: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.gcn = unit_gcn(in_channels, out_channels, A, **kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.gcn(x) + res
        return self.relu(x)


class GCNModel(nn.Module):
    """STGCN backbone.

    Spatial Temporal Graph Convolutional
    Networks for Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1801.07455>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'VC'``.
        ch_ratio (int): Inflation ratio of the number of channels.
            Defaults to 2.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        stage_cfgs (dict): Extra config dict for each stage.
            Defaults to ``dict()``.

        Examples:
        torch.Size([2, 2, 256, 38, 18])
        torch.Size([2, 2, 256, 38, 25])
        torch.Size([2, 2, 256, 38, 17])
        torch.Size([2, 2, 256, 38, 17])
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'VC',
                 ch_ratio: int = 4,
                 num_person: int = 1,
                 num_stages: int = 3,
                 hidden_size: int = 1152,
                 inflate_stages: List[int] = [2, 3],
                 down_stages: List[int] = [2, 3],
                 dropout_prob: float = 0.1,
                 **kwargs) -> None:
        super().__init__()

        self.graph = Graph(**graph_cfg)
        self.dropout_prob = dropout_prob
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                STGCNBlock(
                    in_channels,
                    base_channels,
                    A.clone(),
                    residual=False,
                    **kwargs)
            ]

        inflate_times = 0
        for i in range(2, num_stages+1):
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio**inflate_times + EPS)
            base_channels = out_channels
            modules.append(
                STGCNBlock(in_channels, out_channels, A.clone(), **kwargs))


        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = torch.nn.ModuleList(modules)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels, hidden_size)
        self.embedding_null = nn.Parameter(torch.randn(hidden_size) * 0.02)


    def forward(self, x: torch.Tensor, is_train: bool = True, wo_condition=False) -> torch.Tensor:
        """Defines the computation performed at every call."""
        use_dropout = self.dropout_prob > 0
        B, V, C = x.size()

        if not wo_condition:
            if self.data_bn_type == 'MVC':
                # x = self.data_bn(x.view(N, M * V * C, T))
                pass
            else:
                x = self.data_bn(x.view(B, V * C))
            x = x.view(B, V, C).permute(0, 2, 1).contiguous()

            for i in range(self.num_stages):
                x = self.gcn[i](x)

            x = self.fc(self.pool(x).squeeze())

            if is_train and use_dropout:
                drop_ids = torch.rand(B, device=x.device) < self.dropout_prob
                x = torch.where(drop_ids.unsqueeze(-1), self.embedding_null, x)  # x.shape=(B,C), self.embedding_null.shape=(C)

        else:
            x = self.embedding_null.unsqueeze(0).repeat(B, 1)  # x.shape=(B,C)

        return x



if __name__ == '__main__':
    from src.utils.train_utils import parse_configs
    import torch
    from src.utils.model_utils import instantiate_from_config
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    def get_diffusion_model(args) -> object:
        model = instantiate_from_config(args)
        if args['ckpt_path'] is not None:
            ckpt = torch.load(args['ckpt_path'], map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
        model.y_embedder = GCNModel(**args['GCN_cfg'])

        return model

    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(r'F:\chexiao\project\RAE\re_flow\config\DiDH_XL_DINOv2_B.yaml')

    model = get_diffusion_model(model_config).to('cuda')
    print('=======')