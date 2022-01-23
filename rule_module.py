# -*- coding: utf-8 -*-
"""
Created on 2022/1/23 18:04
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd
import numpy as np

from init_config import src_dir, data_dir
from data_process import data_process, qa_type_rule


def rule_for_qa(dataset):
    qa_df = dataset['qa_data']
    recipes = dataset['recipe_data']
    return dataset


if __name__ == '__main__':
    dataset_model_vali, dataset_rule_vali = data_process('vali')
    rule_result = rule_for_qa(dataset_rule_vali)
    print('END')
