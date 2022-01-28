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
import re
import copy

from init_config import src_dir, data_dir
from data_process import data_process
from trainer import extract_qa_manager
from rule_module import rule_for_qa


# 整理预测结果为提交格式
def convert_pred_result_to_submission_format(pred_result):
    r2vq_pred_result = {}
    for recipe_id, tmp_df in pred_result.groupby(['recipe_id']):
        single_recipe_submission = {}
        for idx, row in tmp_df.iterrows():
            single_recipe_submission[row['question_id']] = row['pred_answer']
        r2vq_pred_result[recipe_id] = single_recipe_submission
    return r2vq_pred_result


# 解析log文件
def analyze_log(filename=None):
    filenames = [
        'log_0122_v2.1.0_f1-86.log'
        , 'log_0123_v2.1.1_f1-86.log'
        , 'log_0123_v2.1.2_f1-86.log'
        , 'log_0123_v2.1.3_f1-86.log'
        , 'log_0123_v2.1.4_f1-86.log'
        , 'log_0123_v2.1.5_f1-86.log'
        , 'log_0123_v2.1.6_f1-86.log'
        , 'log_0123_v2.1.7_f1-86.log'
        , 'log_0123_v2.1.8_f1-86.log'
        , 'log_0124_v2.1.9_f1-86.log'
        , 'log_0124_v2.1.10_f1-86.log'
    ]

    log_dfs = []
    for filename in filenames:
        file_path = os.path.join(data_dir, 'log', filename)

        eval_log = []
        seed = None
        first_or_last_flag = None
        upos_flag = None
        entity_flag = None
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\'', '\"')

                if re.match(r'--seed (?P<keyword>.+) \\', line):
                    seed = re.match(r'--seed (?P<keyword>.+) \\', line).group(1)
                if re.match(r'--embed_at_first_or_last (?P<keyword>.+) \\', line):
                    first_or_last = re.match(r'--embed_at_first_or_last (?P<keyword>.+) \\', line).group(1)
                if re.match(r'--use_upos (?P<keyword>.+) \\', line):
                    use = re.match(r'--use_upos (?P<keyword>.+) \\', line).group(1)
                    upos_flag = 'use_upos' if use == 'True' else 'no_upos'
                if re.match(r'--use_entity (?P<keyword>.+) \\', line):
                    use = re.match(r'--use_entity (?P<keyword>.+) \\', line).group(1)
                    entity_flag = 'use_entity' if use == 'True' else 'no_entity'

                try:
                    info = json.loads(line)
                except:
                    info = {}
                if isinstance(info, dict) and 'eval_f1' in info.keys():
                    eval_log.append(info)

        # log_df = pd.DataFrame(eval_log).set_index('epoch')
        log_df = pd.DataFrame(eval_log)
        log_df['seed'] = seed
        log_df['first_or_last'] = first_or_last
        log_df['upos'] = upos_flag
        log_df['entity'] = entity_flag

        log_dfs.append(log_df)

    all_log_df = pd.concat(log_dfs)
    # all_log_df.to_csv(os.path.join(data_dir, 'log', filename.replace('.log', '.csv')), sep=',', encoding='gbk')
    all_log_df.to_csv(os.path.join(data_dir, 'log', 'log_v2.1.csv'), sep=',', encoding='gbk')


if __name__ == "__main__":
    # analyze_log()
    print("BEGIN")

    # 准备数据
    # if False:
    if os.path.exists(u'D:'):
        dataset_model_vali, dataset_rule_vali = data_process('vali')
        dataset_model_vali = {key: value[:2] for key, value in dataset_model_vali.items()}
        dataset_model_vali = Dataset.from_dict(dataset_model_vali)
        # dataset_model_test = dataset_model_vali
        dataset_model_test, dataset_rule_test = data_process('test')
        dataset_model_test = {key: value for key, value in dataset_model_test.items()}
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

    # 获取规则结果
    rule_pred_result = rule_for_qa(dataset_rule_test)
    rule_pred_result['pred_answer'] = rule_pred_result['pred_answer'].apply(
        lambda x: None if (x == 'N/A' or x == '') else None)
    # 获取模型结果
    model_pred_result = extract_qa_manager(datasets_model)
    model_pred_result['pred_answer'] = model_pred_result['pred_answer'].apply(
        lambda x: None if (x == 'N/A' or x == '') else None)
    # 汇总规则和模型的结果
    used_cols = ['recipe_id', 'question_id', 'question', 'pred_answer', 'answer', 'qa_type']
    pred_result = pd.concat([rule_pred_result[used_cols], model_pred_result[used_cols]])

    # 整理预测结果为提交格式
    r2vq_pred_result = convert_pred_result_to_submission_format(pred_result)

    # 生成提交文件
    submit_filename = 'local_r2vq_pred.json' if os.path.exists(u'D:') else 'r2vq_pred.json'
    with open(os.path.join(src_dir, submit_filename), 'w', encoding='utf-8') as f:
        json.dump(r2vq_pred_result, f)

    # 生成不带规则模块输出的提交文件
    none_rule_pred_result = copy.deepcopy(rule_pred_result)
    none_rule_pred_result['pred_answer'] = ''
    no_rule_pred_result = pd.concat([none_rule_pred_result[used_cols], model_pred_result[used_cols]])
    no_rule_r2vq_pred_result = convert_pred_result_to_submission_format(no_rule_pred_result)
    no_rule_submit_filename = 'local_norule_r2vq_pred.json' if os.path.exists(u'D:') else 'norule_r2vq_pred.json'
    with open(os.path.join(src_dir, no_rule_submit_filename), 'w', encoding='utf-8') as f:
        json.dump(no_rule_r2vq_pred_result, f)

    # 测试用
    if os.path.exists(u'D:'):
        pred_result['family_id'] = pred_result['question_id'].apply(lambda x: x.split('-')[0])
        # pred_result['pred_answer'] = None
        pred_result['pred_answer'] = pred_result['family_id'].apply(lambda x: None if x == '18' else '')

    print('END\n')
