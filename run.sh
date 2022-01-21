python task9_main.py \
--model_name_or_path /media/archfool/data/data/huggingface/bert-base-uncased/ \
--output_dir /media/archfool/data/data/SemEval-2022/task9/result0120_v1.8/ \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--gradient_accumulation_steps 4 \
--seed 1248 \
--disable_tqdm False \
--max_seq_length 512 \
--doc_stride 128 \
--overwrite_output_dir True \
--save_strategy epoch \
--save_total_limit 5 \
--evaluation_strategy epoch \
--dataloader_num_workers 0 \
--max_steps -1 \
--logging_first_step True


# nohup test.py > /media/archfool/data/data/SemEval-2022/task9/log/runoob.log 2>&1 &

#--resume_from_checkpoint /media/archfool/data/data/SemEval-2022/task9/tmp/checkpoint-12425 \
#--max_train_samples 10 \
#--max_eval_samples 10 \

#--resume_from_checkpoint /media/archfool/data/data/SemEval-2022/task9/tmp/checkpoint-12425

#--log_level
#--logging_dir D:\data\tmp\qa\runs\Jan12_09-20-06_LAPTOP-J1U2PL5V
#--logging_first_step True
#--logging_strategy epoch
#--logging_strategy steps
#--logging_steps 500

#--save_strategy epoch
#--save_strategy steps
#--save_steps 500
#--save_total_limit 5

#--evaluation_strategy epoch
#--evaluation_strategy steps
#--eval_steps 1000


