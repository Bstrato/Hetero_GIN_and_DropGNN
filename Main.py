#adopted from https://github.com/KarolisMart/DropGNN
import os
import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GINConv
from torch_geometric.loader import NeighborSampler
from sklearn.model_selection import StratifiedKFold
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from torch_geometric.data import HeteroData
from Models import Models

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'





def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file exists.")
    data = torch.load(data_path)
    if isinstance(data, tuple):
        data = data[0]
    if not isinstance(data, HeteroData):
        raise ValueError(f"Expected HeteroData object, but got {type(data)}")
    return data

def prepare_data(data, device):
    data = data.to(device)
    # Add self-loops if they don't exist
    for node_type in data.node_types:
        if (node_type, 'to', node_type) not in data.edge_types:
            num_nodes = data[node_type].num_nodes
            data[node_type, 'to', node_type].edge_index = torch.arange(num_nodes, device=device).repeat(2, 1)
    return data

def get_model_inputs(data):
    in_channels_dict = {}
    num_nodes_dict = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            in_channels_dict[node_type] = data[node_type].x.size(1)
        else:
            in_channels_dict[node_type] = 0
        num_nodes_dict[node_type] = data[node_type].num_nodes

    metadata = (data.node_types, data.edge_types)
    return in_channels_dict, num_nodes_dict, metadata

def separate_data(num_nodes, seed=0):
    indices = torch.randperm(num_nodes)
    train_idx = indices[:int(0.6 * num_nodes)]
    val_idx = indices[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    test_idx = indices[int(0.8 * num_nodes):]
    return train_idx, val_idx, test_idx

def train(model, optimizer, x_dict, edge_index_dict, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(x_dict, edge_index_dict)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validation(model, x_dict, edge_index_dict, y, mask):
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        pred = out[mask].argmax(dim=-1)
        correct = pred.eq(y[mask]).sum().item()
        acc = correct / mask.sum().item()
        precision = precision_score(y[mask].cpu(), pred.cpu(), average='macro')
        recall = recall_score(y[mask].cpu(), pred.cpu(), average='macro')
    return acc, precision, recall


def plot_metrics(metrics, args):
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_loss'], label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    loss_filename = f"plots/DBLP_{'DropGNN' if args.drop_gnn else 'GIN'}_loss_plot.png"
    plt.savefig(loss_filename)
    plt.close()
    print(f"Loss plot saved as {loss_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_acc'], label='Training Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    acc_filename = f"plots/DBLP_{'DropGNN' if args.drop_gnn else 'GIN'}_accuracy_plot.png"
    plt.savefig(acc_filename)
    plt.close()
    print(f"Accuracy plot saved as {acc_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_precision'], label='Training Precision')
    plt.plot(epochs, metrics['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    precision_filename = f"plots/DBLP_{'DropGNN' if args.drop_gnn else 'GIN'}_precision_plot.png"
    plt.savefig(precision_filename)
    plt.close()
    print(f"Precision plot saved as {precision_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_recall'], label='Training Recall')
    plt.plot(epochs, metrics['val_recall'], label='Validation Recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.tight_layout()
    recall_filename = f"plots/DBLP_{'DropGNN' if args.drop_gnn else 'GIN'}_recall_plot.png"
    plt.savefig(recall_filename)
    plt.close()
    print(f"Recall plot saved as {recall_filename}")


def plot_class_distribution(data):
    y = data['author'].y.cpu()
    class_counts = torch.bincount(y)
    num_classes = len(class_counts)

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), class_counts)
    plt.title('Distribution of Classes in DBLP Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.xticks(range(num_classes))
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    plt.close()

    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count.item()} instances")


def update_metrics(metrics, loss, train_acc, val_acc, train_precision, val_precision, train_recall, val_recall):
    metrics['train_loss'].append(loss)
    metrics['val_loss'].append(loss)
    metrics['train_acc'].append(train_acc)
    metrics['val_acc'].append(val_acc)
    metrics['train_precision'].append(train_precision)
    metrics['val_precision'].append(val_precision)
    metrics['train_recall'].append(train_recall)
    metrics['val_recall'].append(val_recall)



def main(args):
    print(args, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    data = load_data(args.data_path)
    data = prepare_data(data, device)

    print("Type of data:", type(data))
    print("Content of data:", data)

    # Print edge index shapes for debugging
    for edge_type, edge_index in data.edge_index_dict.items():
        print(f"Edge type: {edge_type}, Edge index shape: {edge_index.shape}")

    plot_class_distribution(data)

    num_classes = data['author'].y.max().item() + 1
    print(f"Number of classes: {num_classes}")

    num_nodes = data['author'].num_nodes
    train_idx, val_idx, test_idx = separate_data(num_nodes)

    if 'train_mask' in data['author']:
        train_mask = data['author'].train_mask
        val_mask = data['author'].val_mask
        test_mask = data['author'].test_mask
    else:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask[val_idx] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask[test_idx] = True

    in_channels_dict, num_nodes_dict, metadata = get_model_inputs(data)

    # Model
    if args.drop_gnn:
        model = Models.DropHeteroGIN(metadata, args.hidden_units, 3, num_classes, in_channels_dict, args.dropout,
                              num_runs=10).to(device)
        model.num_nodes_dict = num_nodes_dict
    else:
        model = Models.HeteroGIN(metadata, args.hidden_units, 3, num_classes, in_channels_dict).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    metrics = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': []
    }

    best_val_acc = 0
    final_test_acc = 0
    final_test_precision = 0
    final_test_recall = 0

    for epoch in tqdm(range(1, 201)):
        loss = train(model, optimizer, data.x_dict, data.edge_index_dict, data['author'].y, train_mask)
        train_acc, train_precision, train_recall = validation(model, data.x_dict, data.edge_index_dict,
                                                              data['author'].y, train_mask)
        val_acc, val_precision, val_recall = validation(model, data.x_dict, data.edge_index_dict, data['author'].y,
                                                        val_mask)

        scheduler.step()

        update_metrics(metrics, loss, train_acc, val_acc, train_precision, val_precision, train_recall, val_recall)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc, final_test_precision, final_test_recall = validation(model, data.x_dict,
                                                                                 data.edge_index_dict, data['author'].y,
                                                                                 test_mask)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, '
              f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}')

    print(f'Final Test Accuracy: {final_test_acc:.4f}')
    print(f'Final Test Precision: {final_test_precision:.4f}')
    print(f'Final Test Recall: {final_test_recall:.4f}')

    plot_metrics(metrics, args)

if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dropout', type=float, default=0.5, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 128])
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[16, 32, 64])
    parser.add_argument('--drop_gnn', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='./data_hetero_dblp/processed/hetero_dblp_data.pt')

    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '24:00:00'
        cluster.memory_mb_per_node = '12G'
        job_name = f'DBLP{"_DropGNN" if args.drop_gnn else ""}'
        cluster.per_experiment_nb_cpus = 2
        cluster.per_experiment_nb_gpus = 1
        cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name='DBLP')
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)