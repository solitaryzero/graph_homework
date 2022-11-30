from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import DataLoader

dataset = NodePropPredDataset(name = "ogbn-arxiv")

 
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0]