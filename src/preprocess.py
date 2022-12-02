from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


from logger import Logger

from transformers import AutoModel

dataset_path = "./dataset/ogbn_arxiv/"

def read_map(base_path):
    title2abs_file = os.path.join(base_path, 'mapping/titleabs.tsv')
    title_df = pd.read_csv(title2abs_file, sep='\t', header=None, names=['paper id', 'title', 'abstract'])
    title_df = title_df.dropna()
    title_df[['paper id']] = title_df[['paper id']].astype(np.int64)

    idx2id_file = os.path.join(base_path, 'mapping/nodeidx2paperid.csv.gz')
    id_df = pd.read_csv(idx2id_file, compression='gzip')
    
    return title_df, id_df

def write_to_json(split, idx_list, label_map, node_abstract):

    split_list = list()
    for idx in idx_list:
        item = dict({
            "title": node_abstract[idx][0].replace("\n","").strip(),
            "abstract": node_abstract[idx][1].replace("\n","").strip(),
            "label": label_map[idx].squeeze().item()
        })
        split_list.append(item)
    
    print(f"#lines of {split}: {len(split_list)}")

    with open(os.path.join(dataset_path,"cls",f"{split}.json"), 'w', encoding="utf-8") as fout:
        for line in split_list:
            fout.write(f"{json.dumps(line)}\n")
    
    print(f"Finished writing {split} file")

    return


def main():
    
    print("Load Dataset")

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())
    
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_split_label = data.y[split_idx['train']].squeeze().tolist()
    valid_split_label = data.y[split_idx['valid']].squeeze().tolist()
    test_split_label = data.y[split_idx['test']].squeeze().tolist()

    train_split_idx = split_idx['train'].squeeze().tolist()
    valid_split_idx = split_idx['valid'].squeeze().tolist()
    test_split_idx = split_idx['test'].squeeze().tolist()

    title_map, id_map = read_map(base_path=dataset_path)
    node2abstract = {}
    print("Start processing")
    for pid in tqdm(id_map['paper id']):
        abstract_row = title_map[title_map['paper id'] == pid]

        title, abstract = abstract_row['title'], abstract_row['abstract']
        nid = id_map[id_map['paper id'] == pid]['node idx']
        nid, title, abstract = nid.item(), title.item(), abstract.item()

        node2abstract[nid] = (title, abstract)
    
    write_to_json("train", train_split_idx, data.y, node2abstract)
    write_to_json("valid", valid_split_idx, data.y, node2abstract)
    write_to_json("test", test_split_idx, data.y, node2abstract)
    

if __name__ == "__main__":
    main()