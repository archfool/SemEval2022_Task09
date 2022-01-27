nohup \
python task9_main.py \
--model_name_or_path /media/archfool/data/data/huggingface/bert-large-uncased/ \
--output_dir /media/archfool/data/data/SemEval-2022/task9/result0127_v3.0.5/ \
--dataset_name squad \
--do_train True \
--do_eval True \
--do_predict True \
--embed_at_first_or_last first \
--use_upos True \
--use_entity True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--learning_rate 3e-5 \
--num_train_epochs 20 \
--gradient_accumulation_steps 1 \
--seed 4321 \
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

可以将操作的核心动词作为锚点。
entity列和hidden列是一组。在entity列被标记为EVENT。hidden列仅在entity=EVENT时，可能存在值。
upos列和argX列事一组。在upos列被标为VERB或其它。argX列仅在upos=VERB时，可能存在值。每列argX，有且仅有一个核心动词V。
entity=EVENT和upos=VERB存在一定的共现性。


