# CUDA_VISIBLE_DEVICES=0 python ./src/bert_gcn.py \
#     --bert_embedding_path ./dataset/ogbn_arxiv_bert \
#     --bert_dim 768 \
#     --use_sage

# CUDA_VISIBLE_DEVICES=0 python ./src/bert_gcn.py \
#     --bert_embedding_path ./dataset/ogbn_arxiv_sentbert \
#     --bert_dim 768 \
#     --use_sage

CUDA_VISIBLE_DEVICES=0 python ./src/bert_gcn.py \
    --bert_embedding_path ./dataset/ogbn_arxiv_sentbert_titleonly \
    --bert_dim 768 