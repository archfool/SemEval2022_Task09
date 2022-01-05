# -*- coding: utf-8 -*-
"""
Created on 2021/12/28 16:02
author: ruanzhihao_archfool
"""

import os
import pandas as pd
import numpy as np
from conllu import parse
from init_config import src_dir, data_dir


def bak211228():
    data_val_dir = os.path.join(data_dir, 'val')

    with open(os.path.join(data_val_dir, 'crl_srl.csv'), 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            # 跳过空行
            if '' == line or '\n' == line:
                line = f.readline()
                continue

            # 根据行首是否以#开始，决定以何种方式分列
            if '#' == line[0]:
                line = line.split(' ')
            else:
                line = line[0].split('\t')

            # 以#开始的行
            if '#' == line[0]:
                pass
            # 以数字开始的行
            elif line[0].isdigit():
                pass
            # 其它类型
            else:
                print(line)
            # 读取下一行
            line = f.readline()


"""
CONLL标注格式包含10列，分别为：
———————————————————————————
ID   FORM    LEMMA   CPOSTAG POSTAG  FEATS   HEAD    DEPREL  PHEAD   PDEPREL
———————————————————————————

前８列，其含义分别为：
1    ID      当前词在句子中的序号，１开始.
2    FORM    当前词语或标点
3    LEMMA   当前词语（或标点）的原型或词干，在中文中，此列与FORM相同
4    CPOSTAG 当前词语的词性（粗粒度）
5    POSTAG  当前词语的词性（细粒度）
6    FEATS   句法特征，在本次评测中，此列未被使用，全部以下划线代替。
7    HEAD    当前词语的中心词
8    DEPREL  当前词语与中心词的依存关系

在CONLL格式中，每个词语占一行，无值列用下划线'_'代替，列的分隔符为制表符'\t'，行的分隔符为换行符'\n'；句子与句子之间用空行分隔。
"""

"""
1.ID: Word index, integer starting at 1 for each new sentence; .
2.FORM: Word form or punctuation symbol.
3.LEMMA: Lemma of word form.
4.UPOS: Universal POS tag.
5.ENTITY: Cooking entities of types: EVENT, HABITAT, TOOL, EXPLICITINGREDIENT and IMPLICITINGREDIENT.
6.PART: Word index of the head of EVENT entity when the participant-of relation exists between the event and another entity in current line.
7.PART: Word index of the head of EVENT entity when the result-of relation exists between the event and another entity in current line.
8.HIDDEN: Hidden entities that are involved in the event in current line; see below for details.
9.COREF: Coreference ID for entities that are cross-referred. It is represented as LEMMA.step_id.sent_id.token.id, e.g. asparagus.2.1.3.
10.PREDICATE: The sense of the word that is annotated as a predicate.
11.ARG1: The arguments of the first predicate in current sentence.
12-20. ARGX: The arguments of the X-th predicate in current sentence.
"""

"""
We adopt the concept of "question families" as outlined in the CLEVR dataset (Johnson et al., 2017). While some 
question families naturally transfer over from the VQA domain (e.g., integer comparison, counting), other concepts 
such as ellipsis and object lifespan must be employed to cover the full extent of competency within procedural 
texts.
"""


def parse_recipe(recipe):
    len(recipe[0].metadata)
    # 提取QA数据
    q_list = []
    a_list = []
    for key, value in recipe[0].metadata.items():
        if key.__contains__('question'):
            q_list.append((key, value))
        elif key.__contains__('answer'):
            a_list.append((key, value))
    assert len(q_list) == len(a_list)
    for key, value in q_list + a_list:
        recipe[0].metadata.pop(key)
    qa_list = []
    for (q_key, q_value), (a_key, a_value) in zip(q_list, a_list):
        q_id = q_key.split(' ')[1]
        a_id = a_key.split(' ')[1]
        assert q_id == a_id
        cat_id, seq_id = q_id.split('-')
        qa_list.append((cat_id, seq_id, q_value, a_value))
    qa_df = pd.DataFrame(qa_list, columns=['cat_id', 'seq_id', 'question', 'answer'])
    qa_df['cat_id'] = qa_df['cat_id'].astype(int)
    qa_df['seq_id'] = qa_df['seq_id'].astype(int)
    qa_df = qa_df.sort_values(['cat_id', 'seq_id']).reset_index(drop=True)

    # 提取元数据
    metadata = {}
    for key, value in recipe[0].metadata.items():
        if key.__contains__('metadata'):
            metadata[key.split(':')[-1]] = value
    for key in metadata.keys():
        recipe[0].metadata.pop('metadata:' + key)
    # 提取菜谱ID
    # metadata['newdoc_id'] = recipe[0].metadata.pop('newdoc id')
    newdoc_id = recipe[0].metadata.pop('newdoc id')
    # 分流食材清单和工序步骤
    ingredients = []
    directions = []
    for sent in recipe:
        if 'sent_id' not in sent.metadata.keys():
            # print(metadata['url'])
            # todo 验证集中的metadata的sent，都是ingredients类型的，有时间的话要对训练集做验证
            ingredients[-1].extend(sent)
            pass
        elif 'ingredients' == sent.metadata['sent_id'].split('::')[1]:
            ingredients.append(sent)
        else:
            directions.append(sent)
    # 原始菜谱的文本长度
    metadata['seq_len'] = sum([len(x) for x in ingredients]) + sum([len(x) for x in directions])
    # 返回新结构的菜谱
    new_recipe = {
        'newdoc_id': newdoc_id,
        'metadata': metadata,
        'qa_df': qa_df,
        'ingredients': ingredients,
        'ingredient_dfs': [pd.DataFrame([x for x in ingredient]) for ingredient in ingredients],
        'directions': directions,
        'direction_dfs': [pd.DataFrame([x for x in direction]) for direction in directions],
    }
    return new_recipe


if __name__ == "__main__":
    data_val_dir = os.path.join(data_dir, 'val')
    data_train_dir = os.path.join(data_dir, 'train')
    data_test_dir = os.path.join(data_dir, 'test')
    # fields = ['id', 'from', 'lemma', 'upos', 'entity', 'part', 'hidden', 'coref', 'predicate', 'arg1']
    fields = None

    # for file in os.listdir(data_val_dir):
    #     print(file)

    with open(os.path.join(data_val_dir, 'crl_srl.csv'), 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.split('\n\n\n')
        data = [parse(d, fields=fields) for d in data]

    recipes = []
    for recipe in data:
        recipes.append(parse_recipe(recipe))
        # for sent in recipe:
        #     for token in sent:
        #         if 10 != len(token):
        #             print(token)
    for recipe in recipes:
        print(recipe['metadata']['seq_len'])
    if True:
        qa_all = pd.concat([r['qa_df'] for r in recipes]).sort_values(['cat_id', 'seq_id']).reset_index(drop=True)
        qa_all.to_csv(os.path.join(data_dir, 'qa_val.csv'), index=None, sep='\t', encoding='utf-8')
    print('END')
