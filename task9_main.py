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


# 解析log文件
def analyze_log(filename):
    # filename = 'log_0122_v2.1.0_f1-86.log'
    # filename = 'log_0123_v2.1.1_f1-86.log'
    # filename = 'log_0123_v2.1.2_f1-86.log'
    # filename = 'log_0123_v2.1.3_f1-86.log'
    # filename = 'log_0123_v2.1.4_f1-86.log'
    filename = 'log_0123_v2.1.5_f1-86.log'
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

    log_df.to_csv(os.path.join(data_dir, filename.replace('log', 'csv')), sep=',', encoding='gbk')


if __name__ == "__main__":
    print("BEGIN")
    # if False:
    if os.path.exists(u'D:'):
        dataset_vali = data_process('vali')
        dataset_vali = {key: value[:2] for key, value in dataset_vali.items()}
        dataset_vali = Dataset.from_dict(dataset_vali)
        # dataset_test = dataset_vali
        dataset_test = data_process('test')
        dataset_test = {key: value[:2] for key, value in dataset_test.items()}
        dataset_test = Dataset.from_dict(dataset_test)
        datasets = {'train': dataset_vali, 'validation': dataset_vali, 'test': dataset_test}
    else:
        dataset_train = data_process('train')
        dataset_train = Dataset.from_dict(dataset_train)
        dataset_vali = data_process('vali')
        dataset_vali = Dataset.from_dict(dataset_vali)
        dataset_test = data_process('test')
        dataset_test = Dataset.from_dict(dataset_test)
        datasets = {'train': dataset_train, 'validation': dataset_vali, 'test': dataset_test}

    extract_qa_manager(datasets)
    print('END')
