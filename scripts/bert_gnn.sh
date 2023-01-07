# debug_mode="-m debugpy --listen 127.0.0.1:5678 --wait-for-client"
# CUDA_VISIBLE_DEVICES=0 python ./src/bert_gcn.py \
#     --bert_embedding_path ./dataset/ogbn_arxiv_bert \
#     --bert_dim 768 \
#     --use_sage

# CUDA_VISIBLE_DEVICES=6 python ./src/bert_gcn.py \
#     --bert_embedding_path ./dataset/ogbn_arxiv_sentbert \
#     --bert_dim 768 \
#     --model_name gat

CUDA_VISIBLE_DEVICES=0 python ./src/bert_gcn.py \
    --bert_embedding_path ./dataset/ogbn_arxiv_bertcls \
    --bert_dim 768 \
    --model_name sage


# CUDA_VISIBLE_DEVICES=4 python ${debug_mode} ./src/bert_gcn.py \
#     --bert_embedding_path ./dataset/ogbn_arxiv_sentbert_titleonly \
#     --bert_dim 768 