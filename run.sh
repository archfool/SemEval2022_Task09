nohup \
python task9_main.py \
--model_name_or_path /media/archfool/data/data/SemEval-2022/task9/result0128_v3.0.10/checkpoint-18610 \
--output_dir /media/archfool/data/data/SemEval-2022/task9/result0130_v3.0.11/ \
--dataset_name squad \
--do_train false \
--do_eval false \
--do_predict True \
--embed_at_first_or_last first \
--use_upos True \
--use_entity True \
--learning_rate 3e-5 \
--weight_decay 0 \
--num_train_epochs 15 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--seed 1248 \
--disable_tqdm False \
--max_seq_length 512 \
--doc_stride 128 \
--overwrite_output_dir True \
--save_strategy epoch \
--save_total_limit 10 \
--evaluation_strategy epoch \
--dataloader_num_workers 0 \
--max_steps -1 \
--logging_first_step True \
 >> /media/archfool/data/data/SemEval-2022/task9/log/runoob.log 2>&1 &


--model_name_or_path /media/archfool/data/data/SemEval-2022/task9/result0128_v3.0.10/checkpoint-18610 \
--model_name_or_path /media/archfool/data/data/huggingface/bert-large-uncased/ \
--model_name_or_path /media/archfool/data/data/huggingface/roberta-large/ \
--resume_from_checkpoint /media/archfool/data/data/SemEval-2022/task9/result0128_v3.0.10/checkpoint-18610 \

#--resume_from_checkpoint /media/archfool/data/data/SemEval-2022/task9/result0120_v2.0/checkpoint-28290 \
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

local python task9_main.py
--model_name_or_path
D:\data\huggingface\bert-base-uncased
--output_dir
D:\data\SemEval-2022\task9\model_output
--dataset_name
squad
--do_train
True
--do_eval
True
--do_predict
True
--per_device_train_batch_size
2
--per_device_eval_batch_size
2
--learning_rate
3e-5
--num_train_epochs
3
--gradient_accumulation_steps
2
--seed
1234
--disable_tqdm
False
--max_seq_length
99
--doc_stride
32
--max_train_samples
20
--max_eval_samples
20
--overwrite_output_dir
--save_strategy
epoch
--save_total_limit
3
--evaluation_strategy
epoch
--dataloader_num_workers
0
--max_steps
-1
--use_upos
False
--use_entity
True
--embed_at_first_or_last
last

local python rule_module.py
--model_name_or_path
D:\data\huggingface\bert-base-uncased
--output_dir
D:\data\tmp\qa
--dataset_name
squad
--do_train
--do_eval
--do_predict
--per_device_train_batch_size
2
--per_device_eval_batch_size
2
--learning_rate
3e-5
--num_train_epochs
3
--gradient_accumulation_steps
2
--seed
1234
--disable_tqdm
False
--max_seq_length
99
--doc_stride
32
--max_train_samples
20
--max_eval_samples
20
--overwrite_output_dir
--save_strategy
epoch
--save_total_limit
3
--evaluation_strategy
epoch
--dataloader_num_workers
0
--max_steps
-1
--use_upos
False
--use_entity
True
--embed_at_first_or_last
last

