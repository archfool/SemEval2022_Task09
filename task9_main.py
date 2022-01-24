# -*- coding: utf-8 -*-
"""
Created on 2022/1/9 21:40
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd
from datasets import load_dataset, load_metric, Dataset
import json

from init_config import src_dir, data_dir
from data_process import data_process
from trainer import extract_qa_manager
from rule_module import rule_for_qa


# 解析log文件
def analyze_log(filename):
    # filename = 'log_0122_v2.1.0_f1-86.log'
    # filename = 'log_0123_v2.1.1_f1-86.log'
    # filename = 'log_0123_v2.1.2_f1-86.log'
    # filename = 'log_0123_v2.1.3_f1-86.log'
    # filename = 'log_0123_v2.1.4_f1-86.log'
    # filename = 'log_0123_v2.1.5_f1-86.log'
    # filename = 'log_0123_v2.1.6_f1-86.log'
    # filename = 'log_0123_v2.1.7_f1-86.log'
    filename = 'log_0123_v2.1.8_f1-86.log'
    file_path = os.path.join(data_dir, filename)

    eval_log = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\'', '\"')
            try:
                info = json.loads(line)
            except:
                info = {}
            if isinstance(info, dict) and 'eval_f1' in info.keys():
                eval_log.append(info)
    log_df = pd.DataFrame(eval_log).set_index('epoch')

    # log_df.to_csv(os.path.join(data_dir, filename.replace('log', 'csv')), sep=',', encoding='gbk')


if __name__ == "__main__":
    print("BEGIN")
    if False:
    # if os.path.exists(u'D:'):
        dataset_model_vali, dataset_rule_vali = data_process('vali')
        dataset_model_vali = {key: value[:20] for key, value in dataset_model_vali.items()}
        dataset_model_vali = Dataset.from_dict(dataset_model_vali)
        # dataset_model_test = dataset_model_vali
        dataset_model_test, dataset_rule_test = data_process('test')
        dataset_model_test = {key: value[:2] for key, value in dataset_model_test.items()}
        dataset_model_test = Dataset.from_dict(dataset_model_test)
        datasets_model = {'train': dataset_model_vali, 'validation': dataset_model_vali, 'test': dataset_model_test}
    else:
        dataset_model_train, dataset_rule_train = data_process('train')
        dataset_model_train = Dataset.from_dict(dataset_model_train)
        dataset_model_vali, dataset_rule_vali = data_process('vali')
        dataset_model_vali = Dataset.from_dict(dataset_model_vali)
        dataset_model_test, dataset_rule_test = data_process('test')
        dataset_model_test = Dataset.from_dict(dataset_model_test)
        datasets_model = {'train': dataset_model_train, 'validation': dataset_model_vali, 'test': dataset_model_test}

    rule_pred_result = rule_for_qa(dataset_rule_test)
    model_pred_result = extract_qa_manager(datasets_model)
    used_cols = ['recipe_id', 'question_id', 'question', 'pred_answer', 'answer', 'qa_type']
    pred_result = pd.concat([rule_pred_result[used_cols], model_pred_result[used_cols]])

    if True:
    # if not os.path.exists(u'D:'):
        pred_result['family_id'] = pred_result['question_id'].apply(lambda x: x.split('-')[1])
        pred_result['pred_answer'] = ''

        r2vq_pred_result = {}
        for recipe_id, tmp_df in pred_result.groupby(['recipe_id']):
            single_recipe_submission = {}
            for idx, row in tmp_df.iterrows():
                single_recipe_submission[row['question_id']] = row['pred_answer']
            r2vq_pred_result[recipe_id] = single_recipe_submission

        with open(os.path.join(src_dir, 'r2vq_pred.txt'), 'w', encoding='utf-8') as f:
            json.dump(r2vq_pred_result, f)

    print('END')
