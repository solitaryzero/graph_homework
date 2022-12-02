path_to_dataset=dataset/ogbn_arxiv/cls/
path_to_output=./results/bert_cls
log_path=${path_to_output}/log.txt
debug_mode="-m debugpy --listen 127.0.0.1:6673 --wait-for-client"
CUDA_VISIBLE_DEVICES=7 accelerate launch --multi_gpu --mixed_precision=fp16 ${debug_mode} ./src/bert_for_cls.py \
    --model_name_or_path bert-base-uncased \
    --train_file ${path_to_dataset}/train.json \
    --valid_file ${path_to_dataset}/test.json \
    --max_predict_samples 1000 \
    --do_eval once \
    --preprocessing_num_workers 4 \
    --output_dir ${path_to_output} \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --weight_decay 0.01 \
    --num_train_epochs 6 \
    --with_tracking \
    --report_to wandb \
    --checkpointing_steps epoch 2>&1 | tee -a ${log_path}
    # --resume_from_checkpoint ${path_to_output}/epoch_5 \
    # --do_train \