# code adopted from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/dblp.html#DBLP

import os
import sys
from typing import Callable, List, Optional

import numpy as np
import torch
import scipy.sparse as sp
from itertools import product

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class DBLP(InMemoryDataset):
    url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
    ):
        self.root = root
        # Create necessary directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        if not self._check_exists() or force_reload:
            self.download()
            self._process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    def _check_exists(self) -> bool:
        return all(os.path.exists(f) for f in self.raw_paths) and os.path.exists(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'hetero_dblp_data.pt'

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

    def download(self):
        if self._check_exists():
            print('Files already downloaded and verified')
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)
        print('Download completed successfully')

    def _process(self):
        data = HeteroData()

        node_types = ['author', 'paper', 'term', 'conference']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(os.path.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        x = np.load(os.path.join(self.raw_dir, 'features_2.npy'))
        data['term'].x = torch.from_numpy(x).to(torch.float)

        node_type_idx = np.load(os.path.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        data['conference'].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(os.path.join(self.raw_dir, 'labels.npy'))
        data['author'].y = torch.from_numpy(y).to(torch.long)

        split = np.load(os.path.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['author'][f'{name}_mask'] = mask

        s = {}
        N_a, N_p, N_t, N_c = [data[t].num_nodes for t in node_types]
        s['author'] = (0, N_a)
        s['paper'] = (N_a, N_a + N_p)
        s['term'] = (N_a + N_p, N_a + N_p + N_t)
        s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

        A = sp.load_npz(os.path.join(self.raw_dir, 'adjM.npz'))
        print(f"Adjacency matrix shape: {A.shape}")

        # Create new edges with correct directions
        edge_types = [
            ('author', 'to', 'paper'),
            ('paper', 'to', 'conference'),
            ('conference', 'to', 'term')
        ]

        for src, _, dst in edge_types:
            if (src, dst) == ('conference', 'term'):
                # The original graph does not have an edge from "conference" to "terms". To create this directed edge,
                # the approach here is adopted to faciliate the transformation of the graph.
                A_paper_conf = A[s['paper'][0]:s['paper'][1], s['conference'][0]:s['conference'][1]]
                A_paper_term = A[s['paper'][0]:s['paper'][1], s['term'][0]:s['term'][1]]
                A_conf_term = A_paper_conf.T.dot(A_paper_term)
                A_sub = A_conf_term.tocoo()
            else:
                A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()

            print(f"Submatrix for {src}-{dst}: shape={A_sub.shape}, nnz={A_sub.nnz}")
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, 'to', dst].edge_index = torch.stack([row, col], dim=0)
            else:
                print(f"No edges found for {src}-{dst}")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print('Processing completed successfully')

    def process(self):
        self._process()

    def print_statistics(self):
        print("DBLP Dataset Statistics:")
        print("------------------------")
        for node_type in self.data.node_types:
            num_nodes = self.data[node_type].num_nodes
            num_features = self.data[node_type].num_features if hasattr(self.data[node_type], 'num_features') else 0
            print(f"{node_type.capitalize()} nodes: {num_nodes}, Features: {num_features}")

        for edge_type in self.data.edge_types:
            num_edges = self.data[edge_type].num_edges
            print(f"{edge_type[0].capitalize()}-{edge_type[2].capitalize()} edges: {num_edges}")

        if 'y' in self.data['author']:
            num_classes = self.data['author'].y.max().item() + 1
            print(f"Number of author classes: {num_classes}")

        for split in ['train', 'val', 'test']:
            if f'{split}_mask' in self.data['author']:
                num_split = self.data['author'][f'{split}_mask'].sum().item()
                print(f"Number of authors in {split} set: {num_split}")

"""
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_hetero_dblp')
    print(f"Dataset will be stored in: {root}")

    try:
        dataset = DBLP(root=root)
        print("Dataset loaded successfully.")

        dataset.print_statistics()

        # This is just an  additional debugging information
        print("\nEdge types in the dataset:")
        for edge_type in dataset.data.edge_types:
            print(f"{edge_type[0]}-{edge_type[2]}: {dataset.data[edge_type].num_edges} edges")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        import traceback

        traceback.print_exc()

print("Script ended")
"""