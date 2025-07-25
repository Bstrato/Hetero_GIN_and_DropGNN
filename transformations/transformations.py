from typing import Optional, Tuple

import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('reverse')
class ReverseEdges(BaseTransform):
    r"""
    Reverses edges in heterogeneous graph.
    Inverse: ReverseEdges
    """
    def __init__(self):
        pass
    def forward(self, data: Data) -> Data:

        for edge_type in data.edge_types:
            data[edge_type].edge_index = data[edge_type].edge_index.flip(0)

        return data