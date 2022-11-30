# CUDA_VISIBLE_DEVICES=0 python ./src/bert_features.py \
#     --dataset_path ./dataset/ogbn_arxiv/ \
#     --output_path ./dataset/ogbn_arxiv_bert \
#     --encoder_model bert-base-uncased

# CUDA_VISIBLE_DEVICES=1 python ./src/bert_features.py \
#     --dataset_path ./dataset/ogbn_arxiv/ \
#     --output_path ./dataset/ogbn_arxiv_sentbert \
#     --encoder_model efederici/sentence-bert-base

CUDA_VISIBLE_DEVICES=1 python ./src/bert_features.py \
    --dataset_path ./dataset/ogbn_arxiv/ \
    --output_path ./dataset/ogbn_arxiv_sentbert_titleonly \
    --encoder_model efederici/sentence-bert-base \
    --title_only