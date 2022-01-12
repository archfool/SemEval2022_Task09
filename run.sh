python task9_main.py \
--model_name_or_path D:\data\huggingface\bert-base-uncased \
--output_dir D:\data\tmp\qa \
--do_train \
--do_eval \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 3e-5 \
--num_train_epochs 50 \
--gradient_accumulation_steps 4 \
--seed 1234 \
--disable_tqdm False \
--max_seq_length 384 \
--doc_stride 128 \
--overwrite_output_dir \
--save_strategy epoch \
--save_total_limit 10 \
--evaluation_strategy epoch \
--dataloader_num_workers 0 \
--max_steps -1

logging_dir=D:\data\tmp\qa\runs\Jan12_09-20-06_LAPTOP-J1U2PL5V,
logging_first_step=False,
logging_steps=500,
logging_strategy=IntervalStrategy.STEPS,

--dataset_name squad \
--max_train_samples 10 \
--max_eval_samples 10 \

### resume_from_checkpoint
### log_level
### logging_dir
--save_strategy epoch
--save_strategy steps
--save_steps 5
--save_total_limit 5
--evaluation_strategy epoch
--evaluation_strategy steps
--eval_steps 10


