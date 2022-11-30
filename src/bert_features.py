import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from transformers import AutoTokenizer, AutoModel


def read_map(base_path):
    title2abs_file = os.path.join(base_path, 'mapping/titleabs.tsv')
    title_df = pd.read_csv(title2abs_file, sep='\t', header=None, names=['paper id', 'title', 'abstract'])
    title_df = title_df.dropna()
    title_df[['paper id']] = title_df[['paper id']].astype(np.int64)
    # print(title_df)
    # input()
    # print(type(title_df['paper id'][0]))

    idx2id_file = os.path.join(base_path, 'mapping/nodeidx2paperid.csv.gz')
    id_df = pd.read_csv(idx2id_file, compression='gzip')
    # print(id_df)
    # print(type(id_df['paper id'][0]))
    
    return title_df, id_df


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode(title, abstract, tokenizer, encoder, sent_bert=False, title_only=False):
    if (title_only):
        text_to_tokenize = title
    else:
        text_to_tokenize = title+abstract
        
    tokenize_results = tokenizer(
        text=text_to_tokenize,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    tokenize_results = tokenize_results.to('cuda')
    with torch.no_grad():
        encode_result = encoder(
            input_ids=tokenize_results['input_ids'],
            attention_mask=tokenize_results['attention_mask'],
            token_type_ids=tokenize_results['token_type_ids']
        )

    if (sent_bert):
        return mean_pooling(encode_result, tokenize_results['attention_mask'])
    else:
        return encode_result['pooler_output'][0].detach().cpu()


def main(args):
    title_map, id_map = read_map(base_path=args.dataset_path)
    node2abstract = {}
    for pid in tqdm(id_map['paper id']):
        abstract_row = title_map[title_map['paper id'] == pid]
        # print(abstract_row)
        # input()
        title, abstract = abstract_row['title'], abstract_row['abstract']
        nid = id_map[id_map['paper id'] == pid]['node idx']
        nid, title, abstract = nid.item(), title.item(), abstract.item()
        # print(nid)
        # print(title)
        # print(abstract)
        # input()
        node2abstract[nid] = (title, abstract)

    node2bert = {}
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    encoder = AutoModel.from_pretrained(args.encoder_model).to('cuda')
    sent_bert = (args.encoder_model == 'efederici/sentence-bert-base')
    for nid in tqdm(node2abstract):
        title, abstract = node2abstract[nid]
        bert_feature = encode(title, abstract, tokenizer, encoder, sent_bert, args.title_only)
        # print(bert_feature)
        # print(bert_feature.shape)
        # input()
        node2bert[nid] = bert_feature

    bert_feature_list = []
    for nid in range(len(node2bert)):
        bert_feature_list.append(node2bert[nid])

    new_x = torch.stack(bert_feature_list)

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    torch.save(new_x, os.path.join(args.output_path, 'bert_embedding.pt'))

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    # # print(dataset.data.x.shape)
    # dataset.data.x = new_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Bert+GNN)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ogbn_arxiv/')
    parser.add_argument('--output_path', type=str, default='./dataset/ogbn_arxiv_bert')
    parser.add_argument('--encoder_model', type=str, default='bert-base-uncased')
    parser.add_argument('--title_only', action='store_true')
    args = parser.parse_args()
    print(args)
    
    main(args)