python task9_main.py \
--model_name_or_path D:\data\huggingface\bert-base-uncased \
--do_train \
--do_eval \
--per_device_train_batch_size 2 \
--learning_rate 3e-5 \
--num_train_epochs 50 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir D:\data\tmp\qa \
--overwrite_output_dir

resume_from_checkpoint
log_level
logging_dir
--save_strategy epoch
--save_strategy steps
--save_steps 5
--save_total_limit 5
--evaluation_strategy steps
--eval_steps 10
--dataloader_num_workers 0
--disable_tqdm True
--gradient_accumulation_steps
4
--per_device_eval_batch_size
4
--save_steps
100
--seed
1234
--evaluation_strategy
epoch

--dataset_name squad \
--max_train_samples 10 \
--max_eval_samples 10 \
