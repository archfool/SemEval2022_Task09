# -*- coding: utf-8 -*-
"""
Created on 2022/1/9 21:40
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd
from datasets import load_dataset, load_metric, Dataset

from data_process import data_process

# sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'framework'))
from DemoExtractQA import extract_qa_manager
from manager import manager
from util_tools import logger
from util_model import ACT2FN

if __name__ == "__main__":
    if False:
    # if os.path.exists(u'D:'):
        dataset_vali = data_process('vali')
        dataset_vali = {key: value[:1] for key, value in dataset_vali.items()}
        dataset_vali = Dataset.from_dict(dataset_vali)
        datasets = {'train': dataset_vali, 'validation': dataset_vali, 'test': dataset_vali}
    else:
        dataset_train = data_process('train')
        dataset_train = Dataset.from_dict(dataset_train)
        dataset_vali = data_process('vali')
        dataset_vali = Dataset.from_dict(dataset_vali)
        datasets = {'train': dataset_train, 'validation': dataset_vali, 'test': dataset_vali}

    extract_qa_manager(datasets)
    print('END')
