root_dir='/media/archfool/data/data'
# nohup \
for embd_pos_flag in first last;
do
  for entity_flag in True False;
  do
    for upos_flag in True False;
    do
      for seed in 1234 4321 1248;
      do
        python test.py \
        --model_name_or_path ${root_dir}/huggingface/bert-base-uncased \
        --output_dir ${root_dir}/SemEval2022/task9/paper/${embd_pos_flag}_entity_${entity_flag}_upos_${upos_flag}/ \
        --dataset_name squad \
        --do_train True \
        --do_eval True \
        --do_predict True \
        --embed_at_first_or_last ${embd_pos_flag} \
        --use_entity ${entity_flag} \
        --use_upos ${upos_flag} \
        --learning_rate 3e-5 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --seed ${seed} \
        --disable_tqdm False \
        --max_seq_length 512 \
        --doc_stride 128 \
        --overwrite_output_dir True \
        --save_strategy epoch \
        --save_total_limit 1 \
        --evaluation_strategy epoch \
        --dataloader_num_workers 0 \
        --max_steps -1 \
        --logging_first_step True \
        ;
        done
    done
  done
done
# >> /media/archfool/data/data/SemEval-2022/task9/log/runoob.log 2>&1 &


