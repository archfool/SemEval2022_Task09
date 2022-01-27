# -*- coding: utf-8 -*-
"""
Created on 2022/1/27 19:27
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd
import numpy as np
from conllu import parse
import json

from init_config import src_dir, data_dir
from data_process import parse_recipe

from pattern import en
from pattern.en import conjugate, lemma, lexeme, SG, PRESENT, PARTICIPLE

# https://stackoverflow.com/questions/3753021/using-nltk-and-wordnet-how-do-i-convert-simple-tense-verb-into-its-present-pas
# pip install pattern
# from pattern.en import *
# print(lemma('gave'))
# give
# print(lexeme('gave'))
# ['give', 'gives', 'giving', 'gave', 'given']
# print(conjugate(verb='give',tense=PRESENT,number=SG)) # he / she / it
# gives

if __name__ == '__main__':

    data_train_dir = os.path.join(data_dir, 'train')
    data_vali_dir = os.path.join(data_dir, 'val')
    data_test_dir = os.path.join(data_dir, 'test')
    fields = ['id', 'form', 'lemma', 'upos', 'entity', 'part1', 'part2', 'hidden', 'coref', 'predicate',
              'arg1', 'arg2', 'arg3', 'arg4', 'arg5', 'arg6', 'arg7', 'arg8', 'arg9', 'arg10']

    recipes = []
    for dataset_name in ['train', 'vali', 'test']:
        if 'train' == dataset_name:
            data_uesd_dir = data_train_dir
        elif 'vali' == dataset_name:
            data_uesd_dir = data_vali_dir
        elif 'test' == dataset_name:
            data_uesd_dir = data_test_dir
        else:
            data_uesd_dir = None

        with open(os.path.join(data_uesd_dir, 'crl_srl.csv'), 'r', encoding='utf-8') as f:
            data = f.read()
            data = data.split('\n\n\n')
            data = [parse(d, fields=fields) for d in data]

        # 读取菜单
        for d in data:
            recipe = parse_recipe(d)
            recipes.append(recipe)
        # recipes = {recipe['newdoc_id']: recipe for recipe in recipes}

    all_direction_df = pd.concat([df for recipe in recipes for df in recipe['direction_dfs']])
    all_direction_df = all_direction_df.drop_duplicates(['form'])
    words = set(all_direction_df['form'].tolist() + all_direction_df['lemma'].tolist())
    present_tense_dict = {}
    for word in words:
        present_tense_dict[word] = conjugate(verb=word, tense=PARTICIPLE, number=SG)

    with open(os.path.join(src_dir, 'present_tense.json'), 'w', encoding='utf-8') as f:
        json.dump(present_tense_dict, f)

    with open(os.path.join(src_dir, 'present_tense.json'), 'r', encoding='utf-8') as f:
        present_tense_map = json.load(f)

    print('END')
