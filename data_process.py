# -*- coding: utf-8 -*-
"""
Created on 2021/12/28 16:02
author: ruanzhihao_archfool
"""

import os
import pandas as pd
import numpy as np
import re
from conllu import parse
import nltk.stem as ns
import copy
import json
from init_config import src_dir, data_dir

lemmer = ns.WordNetLemmatizer()

upos_list = ['other', 'NOUN', 'PUNCT', 'VERB', 'ADP', 'DET', 'CCONJ', 'ADJ', 'ADV', 'NUM', 'PART', 'SCONJ', 'PRON',
             'AUX', 'SYM', 'PROPN', 'X', 'INTJ']
upos_map = {upos: id for id, upos in enumerate(upos_list)}
entity_list = ['other', 'O', 'O-ADD', 'B-EVENT', 'B-EXPLICITINGREDIENT', 'B-ADD-INGREDIENT', 'B-ADD-HABITAT',
               'I-ADD-INGREDIENT', 'I-EXPLICITINGREDIENT', 'B-IMPLICITINGREDIENT', 'B-HABITAT', 'I-ADD-HABITAT',
               'B-ADD-TOOL', 'I-HABITAT', 'I-IMPLICITINGREDIENT', 'B-TOOL', 'I-TOOL', 'I-ADD-TOOL', 'I-EVENT']
entity_map = {entity: id for id, entity in enumerate(entity_list)}

qa_type_rule = ['act_first', 'count', 'place_before_act', 'get_result', 'result_component']
qa_type_model = {
    1: ['act_ref_place', 'act_ref_tool_or_full_act', 'act_ref_igdt'],  # 关键词匹配
    2: ['act_igdt_ref_place', 'act_duration', 'act_extent', 'act_reason', 'act_from_where',
        'act_couple_igdt', 'igdt_amount', 'how_would_you', 'what_do_you'],  # 整句匹配
}

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
SemEval2022_Task9的CONLL标注格式
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
    for key, value in q_list + a_list:
        recipe[0].metadata.pop(key)
    qa_list = []
    if len(q_list) == len(a_list):
        for (q_key, q_value), (a_key, a_value) in zip(q_list, a_list):
            q_id = q_key.split(' ')[1]
            a_id = a_key.split(' ')[1]
            assert q_id == a_id
            family_id, seq_id = q_id.split('-')
            qa_list.append((family_id, seq_id, q_value, a_value))
    elif len(q_list) > 0 and len(a_list) == 0:
        for (q_key, q_value) in q_list:
            q_id = q_key.split(' ')[1]
            family_id, seq_id = q_id.split('-')
            a_value = None
            qa_list.append((family_id, seq_id, q_value, a_value))
    else:
        raise ValueError("len(q_list) and len(a_list) not match!")
    qa_df = pd.DataFrame(qa_list, columns=['family_id', 'seq_id', 'question', 'answer'])
    qa_df['family_id'] = qa_df['family_id'].astype(int)
    qa_df['seq_id'] = qa_df['seq_id'].astype(int)
    qa_df = qa_df.sort_values(['family_id', 'seq_id']).reset_index(drop=True)
    qa_df['question'] = qa_df['question'].apply(lambda x: x.lstrip())
    qa_df['answer'] = qa_df['answer'].apply(lambda x: x.lstrip() if x is not None else x)
    qa_df[['question', 'answer', 'type', 'key_str_q', 'key_str_a']] \
        = qa_df.apply(parse_qa, axis=1, result_type="expand")

    # 提取元数据
    metadata = {}
    for key, value in recipe[0].metadata.items():
        if key.__contains__('metadata'):
            metadata[key.split(':')[-1]] = value
    for key in metadata.keys():
        recipe[0].metadata.pop('metadata:' + key)
    # 提取菜谱ID
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
    ingredient_dfs = [pd.DataFrame([x for x in ingredient]) for ingredient in ingredients]
    direction_dfs = [pd.DataFrame([x for x in direction]) for direction in directions]
    # 根据隐藏角色信息，重写操作步骤文本
    new_direction_dfs = hidden_role_knowledge_enhanced(direction_dfs, ingredient_dfs)
    # 对token字段进行清洗：删去左右空格，小写化
    for tmp_dfs in [ingredient_dfs, direction_dfs, new_direction_dfs]:
        for tmp_df in tmp_dfs:
            tmp_df['form'] = tmp_df['form'].apply(lambda x: x.strip().lower())
            tmp_df['lemma'] = tmp_df['lemma'].apply(lambda x: x.strip().lower())
    # 对食材清单文本的entity字段进行赋值，赋值为【other】
    for tmp_dfs in [ingredient_dfs]:
        for tmp_df in tmp_dfs:
            tmp_df['entity'] = 'other'
    # 对食材清单文本、操作步骤文本的upos字段和entity字段进行ip映射
    for tmp_dfs in [ingredient_dfs, direction_dfs, new_direction_dfs]:
        for tmp_df in tmp_dfs:
            tmp_df['upos_id'] = tmp_df['upos'].map(upos_map)
            tmp_df['entity_id'] = tmp_df['entity'].map(entity_map)

    # 原始菜谱的文本长度
    metadata['seq_len'] = sum([len(x) for x in directions])
    metadata['seq_len_new'] = sum([len(x) for x in new_direction_dfs])

    # 返回新结构的菜谱
    new_recipe = {
        'newdoc_id': newdoc_id,
        'metadata': metadata,
        'qa_df': qa_df,
        'ingredients': ingredients,
        'ingredient_dfs': ingredient_dfs,
        'directions': directions,
        'direction_dfs': direction_dfs,
        'new_direction_dfs': new_direction_dfs,
    }
    return new_recipe


type_family_id_dict = {
    'act_first': [4],
    'count': [0],
    'place_before_act': [17],
    'get_result': [3],
    'result_component': [3],
    'act_ref_igdt': [1],
    'act_ref_place': [2],
    'act_ref_tool_or_full_act': [2] + [5, 6, 8, 10, 14],
    # 'act_ref_tool': [2],
    'act_extent': [5],
    # 'full_act': [5, 6, 8, 10, 14],
    'act_duration': [7],
    # 'add_igdt_place': [8, 12],
    'act_igdt_ref_place': [6, 8, 12, 13],
    'igdt_amount': [9],
    'act_reason': [11, 15],
    'act_couple_igdt': [8, 12, 13],
    'act_from_where': [16],
    'what_do_you': [2, 5, 8, 9, 10],
    'how_would_you': [9],
}
type_q_regex_pattern_dict = {
    'act_first': ['(?P<keyword>.+), which comes first\?'],
    'count': ['How many actions does it take to process the (?P<keyword>.+)\?',
              'How many times is the (?P<keyword>.+) used\?',
              'How many (?P<keyword>.+) are used\?'],
    'place_before_act': ['Where was the (?P<keyword>.+) before (?P<keyword2>.+)\?'],
    'get_result': ['How did you get the (?P<keyword>.+)\?'],
    'result_component': ['What\'s in the (?P<keyword>.+)\?'],
    'act_ref_igdt': ['What should be (?P<keyword>.+)\?'],
    'act_ref_place': ['Where should you (?P<keyword>.+)\?'],
    'act_ref_tool_or_full_act': ['How do you (?P<keyword>.+)\?'],
    # 'act_ref_tool': ['How do you (?P<keyword>.+)\?'],
    'act_extent': ['To what extent do you (?P<keyword>.+)\?'],
    # 'full_act': ['How do you (?P<keyword>.+)\?'],
    'act_duration': ['For how long do you (?P<keyword>.+)\?'],
    # 'add_igdt_place': ['Where do you add (?P<keyword>.+)\?'],
    'act_igdt_ref_place': ['Where do you (?P<keyword>.+)\?'],
    'igdt_amount': ['By how much do you (?P<keyword>.+)\?'],
    'act_reason': ['Why do you (?P<keyword>.+)\?'],
    'act_couple_igdt': ['What do you (?P<keyword>.+) with\?'],
    'act_from_where': ['From where do you (?P<keyword>.+)\?'],
    'what_do_you': ['What do you (?P<keyword>.+)\?'],
    'how_would_you': ['How would you (?P<keyword>.+)\?'],
}
type_a_regex_pattern_dict = {
    'act_first': ['(?P<keyword>.+)'],
    'count': ['(?P<keyword>.+)'],
    'place_before_act': ['(?P<keyword>.+)'],
    'get_result': ['by (?P<keyword>.+)'],
    'result_component': ['(?:the )?(?P<keyword>.+)'],
    'act_ref_igdt': ['(?:the )?(?P<keyword>.+)'],
    'act_ref_place': ['(?P<keyword>.+)'],
    'act_ref_tool_or_full_act': ['(?P<keyword>.+)'],
    # 'act_ref_tool': ['by (?P<keyword>hand)', 'by using a (?P<keyword>.+)'],
    'act_extent': ['(?:until )(?P<keyword>.+)',
                   '(?:till )?(?P<keyword>.+)'],
    # 'full_act': ['(?P<keyword>.+)'],
    'act_duration': ['(?:for )?(?P<keyword>.+)'],
    # 'add_igdt_place': ['(?:to )?(?P<keyword>.+)'],
    'act_igdt_ref_place': ['(?P<keyword>.+)'],
    'igdt_amount': ['(?P<keyword>.+)'],
    'act_reason': ['(?:so )(?P<keyword>.+)',
                   '(?:to )(?P<keyword>.+)',
                   '(?:for )(?P<keyword>.+)',
                   '(?P<keyword>.+)'],
    'act_couple_igdt': ['(?:with )?(?P<keyword>.+)'],
    'act_from_where': ['(?:from )?(?P<keyword>.+)'],
    'what_do_you': ['(?P<keyword>.+)'],
    'how_would_you': ['(?P<keyword>.+)'],
}
type_sep_dict = {
    'act_first': [],
    'count': [],
    'place_before_act': [],
    'get_result': [' and '],
    'result_component': [' and ', ','],
    'act_ref_igdt': [' and ', ','],
    'act_ref_place': [' and ', ','],
    'act_ref_tool_or_full_act': [],
    # 'act_ref_tool': [' and ', ','],
    'act_extent': [' and '],
    # 'full_act': [],
    'act_duration': [],
    # 'add_igdt_place': [],
    'act_igdt_ref_place': [],
    'igdt_amount': [],
    'act_reason': [],
    'act_couple_igdt': [' and ', ','],
    'act_from_where': [],
    'what_do_you': [],
    'how_would_you': [],
}


# 解析QA对
# qa_all[['question', 'answer', 'method', 'type', 'key_str_q', 'key_str_a', 'keyword_a']] = qa_all.apply(parse_qa, axis=1, result_type="expand")
def parse_qa(qa_row):
    question = qa_row['question'].replace(r'\"', "").lstrip()
    answer = qa_row['answer'].replace(r'\"', "").lstrip() if qa_row['answer'] is not None else qa_row['answer']
    family_id = qa_row['family_id']

    # type: # 问答类别标签
    # key_str_q:问题核心文本
    # key_str_a:答案核心文本

    # cat_0：3类 统计操作次数，工具使用次数，食材使用个数
    # cat_1：1类 操作所涉及的食材是什么？
    # cat_2：2类 操作所涉及的场所是什么？操作所涉及的工具是什么？
    # cat_3：2类 某个容器里有什么？如何获得中间食材？
    # cat_4：1类 两个操作里，那个操作更早进行？
    # cat_5：1类 操作执行到什么程度？
    # cat_7：1类 操作执行多长时间？
    # cat_8：1类 在哪里执行操作？ answer的句首大概率要加介词
    # cat_11：1类 执行操作的原因？ answer的句首要加so
    # cat_12：2类 同cat_8，在哪里执行操作？操作的食材是什么和什么？answer的句首要加with
    # cat_15：1类 执行操作的原因？ answer的句首要加to
    # cat_16：1类 执行操作的地方？ answer的句首大概率要加where

    # 遍历所有qa类型
    for qa_type, q_regex_patterns in type_q_regex_pattern_dict.items():
        # 遍历当前qa类型下的所有question正则模板
        for q_regex_pattern in q_regex_patterns:
            regex_q = re.match(q_regex_pattern, question)
            # 匹配到question的正则模板
            if regex_q:
                assert regex_q.group(0) == question
                # 赋值QA类别
                type = qa_type
                # 获取【问题】的关键文本
                key_str_q = regex_q.group(1)

                # answer是None空字段的情况
                if answer is None:
                    key_str_a = None
                    return question, answer, type, key_str_q, key_str_a

                # answer是'NA'字段的情况
                if 'N/A' == answer:
                    assert family_id == 18
                    key_str_a = None
                    return question, answer, type, key_str_q, key_str_a

                # answer不是'NA'字段和None空字段的情况
                else:
                    assert family_id in type_family_id_dict[qa_type]
                    # 遍历当前qa类型下的所有answer正则模板
                    for a_regex_pattern in type_a_regex_pattern_dict[qa_type]:
                        regex_a = re.match(a_regex_pattern, answer)
                        # 匹配到answer的正则模板
                        if regex_a:
                            key_str_a = regex_a.group(1)
                            # keyword_a = [key_str_a]
                            # for sep in type_sep_dict[qa_type]:
                            #     keyword_a = [y.strip() for x in keyword_a for y in x.split(sep)]
                            return question, answer, type, key_str_q, key_str_a

    print(qa_row, flush=True)
    raise ValueError('出现了意料之外的QA模式')


def inter_parse_qa_test(qa_df):
    for idx, qa_row in qa_df.iterrows():
        if qa_row['question'].__contains__("How do you shell the peas in the bowl?"):
            print(qa_row['question'])
        print('{}\t{}'.format(qa_row['question'], qa_row['answer']), flush=True)
        tmp = parse_qa(qa_row)


# 根据隐藏角色信息，重写文本。
def hidden_role_knowledge_enhanced(directions, ingredients):
    def join_role_items(role_items, upos_map, entity):
        role_items_plus = []
        # 标记upos和entity属性
        for role_item in role_items:
            role_item = [[token, upos_map.get(token, 'NOUN'), 'I-ADD-' + entity] for token in role_item]
            role_item[0][2] = 'B-ADD-' + entity
            role_items_plus.append(role_item)
        # 添加连接词
        if 1 == len(role_items_plus):
            new_role_items = role_items_plus[0]
        else:
            new_role_items = []
            for idx in range(len(role_items_plus) - 1):
                new_role_items.extend(role_items_plus[idx])
                new_role_items.append([',', 'PUNCT', 'O-ADD'])
            new_role_items[-1] = ['and', 'CCONJ', 'O-ADD']
            new_role_items.extend(role_items_plus[idx + 1])
        return new_role_items

    # 检测当前文本的最后一个token，是否为标点符号
    def check_last_punct(direction_dfs):
        if 0 == len(direction_dfs):
            return True
        else:
            return 'PUNCT' == direction_dfs[-1]['upos'].iloc[-1]

    upos_map = pd.concat([x for x in ingredients + directions]).set_index(['form'])['upos'].to_dict()
    directions_new = []
    for direction in directions:
        # 新文本
        direction_new = []
        # 遍历所有token
        for idx, row in direction.iterrows():
            # 如果原始文本未能正确分割标点符号，则进行纠正
            re_punct = re.search('[.,;]', row['form'])
            if re_punct and len(row['form']) > 1 and row['form'] not in ['..', '...']:
                span = re_punct.span()
                if span == (0, 1) or span == (len(row['form']) - 1, len(row['form'])):
                    # 生成标点token信息
                    punct_token = row['form'][span[0]: span[1]]
                    punct_token_data = {
                        'id': row['id'],
                        'form': punct_token,
                        'lemma': punct_token,
                        'upos': 'PUNCT',
                        'entity': 'O',
                    }
                    # 判断标点位置
                    if span[0] == 0:
                        row['form'] = row['form'][1:]
                        row['lemma'] = row['lemma'][1:]
                        punct_tokens = [punct_token_data, row.to_dict()]
                    elif span[0] == len(row['form']) - 1:
                        row['form'] = row['form'][:-1]
                        row['lemma'] = row['lemma'][:-1]
                        punct_tokens = [row.to_dict(), punct_token_data]
                    else:
                        punct_tokens = [row.to_dict()]
                        print('token内的标点位置异常:{}'.format(row['form']))
                    token_df = pd.DataFrame(punct_tokens, columns=direction.columns)
                else:
                    token_df = pd.DataFrame([row.to_dict()], columns=direction.columns)
                    print('token内的标点位置异常:{}'.format(row['form']))
            else:
                token_df = pd.DataFrame([row.to_dict()], columns=direction.columns)
            # 处理不含角色信息的token
            if '_' == row['hidden']:
                direction_new.append(token_df)
            # 处理包含角色信息的token
            else:
                cur_token = [token_df]
                hidden_roles = row['hidden'].split('|')
                # 变更hidden_roles的顺序
                hidden_roles_tmp = {'Result': [], 'Habitat': [], 'Tool': [], 'Drop': [], 'Shadow': []}
                for role in hidden_roles:
                    match_flag = False
                    for role_name in hidden_roles_tmp.keys():
                        if role.startswith(role_name):
                            hidden_roles_tmp[role_name].append(role)
                            match_flag = True
                            break
                    if match_flag is False:
                        raise ValueError('出现了意料之外的Hidden Role')
                hidden_roles = hidden_roles_tmp['Result'] + hidden_roles_tmp['Habitat'] + hidden_roles_tmp['Tool'] \
                               + hidden_roles_tmp['Drop'] + hidden_roles_tmp['Shadow']
                # 遍历所有角色
                for role in hidden_roles:
                    # 提取角色名和角色元素
                    role_name, role_items = role.split('=')
                    role_items = [item.split('.')[0].split('_') for item in role_items.split(':')]
                    # Drop, Habitat, Tool, Result, Shadow
                    # 根据不同角色，添加不同的文本
                    if 'Result' == role_name:
                        add_tokens = join_role_items(role_items, upos_map, entity='INGREDIENT')
                        add_tokens = [['to', 'PART', 'O-ADD'], ['get', 'VERB', 'O-ADD']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O-ADD']]
                        if check_last_punct(direction_new) is False:
                            add_tokens = [[',', 'PUNCT', 'O-ADD']] + add_tokens
                        add_tokens_df_data = {
                            'id': [-1 for _ in add_tokens],
                            'form': [x[0] for x in add_tokens],
                            'lemma': [x[0] for x in add_tokens],
                            'upos': [x[1] for x in add_tokens],
                            'entity': [x[2] for x in add_tokens],
                        }
                        add_tokens_df = pd.DataFrame(add_tokens_df_data, columns=direction.columns)
                        direction_new.extend([add_tokens_df] + cur_token)
                        # direction_new = direction_new + [add_tokens_df] + cur_token
                        cur_token = []
                    elif 'Habitat' == role_name:
                        add_tokens = join_role_items(role_items, upos_map, entity='HABITAT')
                        add_tokens = [['in', 'ADP', 'O-ADD']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O-ADD']]
                        if check_last_punct(direction_new + cur_token) is False:
                            add_tokens = [[',', 'PUNCT', 'O-ADD']] + add_tokens
                        add_tokens_df_data = {
                            'id': [-1 for _ in add_tokens],
                            'form': [x[0] for x in add_tokens],
                            'lemma': [x[0] for x in add_tokens],
                            'upos': [x[1] for x in add_tokens],
                            'entity': [x[2] for x in add_tokens],
                        }
                        add_tokens_df = pd.DataFrame(add_tokens_df_data, columns=direction.columns)
                        direction_new.extend(cur_token + [add_tokens_df])
                        # direction_new = direction_new + cur_token + [add_tokens_df]
                        cur_token = []
                    elif 'Tool' == role_name:
                        add_tokens = join_role_items(role_items, upos_map, entity='TOOL')
                        if (1 == len(add_tokens)) and (add_tokens[0][0] in ['hand', 'hands']):
                            add_tokens[0][0] = 'hand'
                            add_tokens = [['by', 'ADP', 'O-ADD']] \
                                         + add_tokens \
                                         + [[',', 'PUNCT', 'O-ADD']]
                        else:
                            add_tokens = [['by', 'ADP', 'O-ADD'], ['using', 'VERB', 'O-ADD'], ['a', 'DET', 'O-ADD']] \
                                         + add_tokens \
                                         + [[',', 'PUNCT', 'O-ADD']]
                        if check_last_punct(direction_new + cur_token) is False:
                            add_tokens = [[',', 'PUNCT', 'O-ADD']] + add_tokens
                        add_tokens_df_data = {
                            'id': [-1 for _ in add_tokens],
                            'form': [x[0] for x in add_tokens],
                            'lemma': [x[0] for x in add_tokens],
                            'upos': [x[1] for x in add_tokens],
                            'entity': [x[2] for x in add_tokens],
                        }
                        add_tokens_df = pd.DataFrame(add_tokens_df_data, columns=direction.columns)
                        direction_new.extend(cur_token + [add_tokens_df])
                        # direction_new = direction_new + cur_token + [add_tokens_df]
                        cur_token = []
                    elif ('Drop' == role_name) or ('Shadow' == role_name):
                        add_tokens = join_role_items(role_items, upos_map, entity='INGREDIENT')
                        add_tokens = [['the', 'DET', 'O-ADD']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O-ADD']]
                        add_tokens_df_data = {
                            'id': [-1 for _ in add_tokens],
                            'form': [x[0] for x in add_tokens],
                            'lemma': [x[0] for x in add_tokens],
                            'upos': [x[1] for x in add_tokens],
                            'entity': [x[2] for x in add_tokens],
                        }
                        add_tokens_df = pd.DataFrame(add_tokens_df_data, columns=direction.columns)
                        direction_new.extend(cur_token + [add_tokens_df])
                        # direction_new = direction_new + cur_token + [add_tokens_df]
                        cur_token = []
                    else:
                        raise ValueError('出现了意料之外的Hidden Role')
        direction_new = pd.concat(direction_new)
        directions_new.append(direction_new)
    return directions_new


def label_single_qa_sample_bak0120(sample, qa, recipe):
    ingredients = recipe['ingredient_dfs']
    new_directions = recipe['new_direction_dfs']
    directions = recipe['direction_dfs']

    """============================模型============================"""
    # 拼接食材清单文本，拼接操作步骤文本。拼接后的文本，剔除了所有空格。
    ingredient_tokens = [token for df in ingredients for token in df['form'].tolist()]
    direction_tokens = [token for df in directions for token in df['form'].tolist()]
    tokens = ingredient_tokens + direction_tokens
    sample['tokens'] = tokens
    text = ''.join(tokens).replace(' ', '')
    # 计算token在文本中的位置偏置（删除了所有空格后的文本）
    offsets = []
    cur_offset = 0
    for token in tokens:
        offsets.append((cur_offset, cur_offset + len(token.replace(' ', ''))))
        cur_offset = offsets[-1][1]

    # 无答案
    if 'N/A' == qa['answer']:
        match_info = 'no_answer'
        sample['label'] = [0 for _ in tokens]
        sample['match_info'] = match_info
        return sample

    # 答案文本完全匹配
    label_offsets = []
    answer = qa['answer'].replace(' ', '').replace('(', '\(').replace(')', '\)')
    for tmp in re.finditer(answer, text):
        label_offsets.append(tmp.span())
    if len(label_offsets) > 0:
        if len(label_offsets) == 1:
            match_info = 'full'
        else:
            # todo 要找到最合适的span，进行标注
            match_info = 'full_duplicate'
        labels = []
        for offset in offsets:
            label = 0
            for label_offset in label_offsets:
                if label_offset[0] <= offset[0] and offset[1] <= label_offset[1]:
                    label = 1
                    continue
            labels.append(label)
        sample['label'] = labels
        sample['match_info'] = match_info
        return sample

    # 答案文本部分匹配
    label_offsets = []
    answer = qa['key_str_a'].replace(' ', '').replace('(', '\(').replace(')', '\)')
    for tmp in re.finditer(answer, text):
        label_offsets.append(tmp.span())
    if len(label_offsets) > 0:
        if len(label_offsets) == 1:
            match_info = 'partial'
        else:
            # todo 要找到最合适的span，进行标注
            match_info = 'partial_duplicate'
        labels = []
        for offset in offsets:
            label = 0
            for label_offset in label_offsets:
                if label_offset[0] <= offset[0] and offset[1] <= label_offset[1]:
                    label = 1
                    continue
            labels.append(label)
        sample['label'] = labels
        sample['match_info'] = match_info
        return sample

    # 答案关键词匹配
    label_offsets = []
    for keyword in qa['keyword_a']:
        kw = keyword.replace(' ', '').replace('(', '\(').replace(')', '\)')
        for tmp in re.finditer(kw, text):
            label_offsets.append(tmp.span())
    if len(label_offsets) > 0:
        match_info = 'keywords'
        # todo 聚合靠得近的关键词
        labels = []
        for offset in offsets:
            label = 0
            for label_offset in label_offsets:
                if label_offset[0] <= offset[0] and offset[1] <= label_offset[1]:
                    label = 1
                    continue
            labels.append(label)
        sample['label'] = labels
        sample['match_info'] = match_info
        return sample


# 提取文本关键词
def get_keywords(str_list, seps, stopwords=[], puncts=[]):
    keywords = str_list
    for punct in puncts:
        keywords = [kw.replace(punct, '') for kw in keywords]

    for sep in seps:
        keywords = [y.strip() for x in keywords for y in x.split(sep)]
        keywords = [kw for kw in keywords if kw not in stopwords]

    return keywords


# 判断target_token（包含多个形态）是否在candidate_tokens中出现
def token_match(target_tokens, candidate_tokens):
    for token in target_tokens:
        if token in candidate_tokens:
            return True
    return False


# 判断target_string是否在context中出现
def string_match(target_string, context_tokens, add_tags=None):
    target_string = target_string.replace(' ', '').replace('(', '\(').replace(')', '\)')
    context_tokens_bak = copy.deepcopy(context_tokens)
    # add_tags表示加入角色信息的标注，-1表示当前位置为添加的角色信息
    # 若add_tags不为None，则从context_tokens中提取出原始文本，进行匹配
    if add_tags is not None:
        context_tokens_tmp = []
        for token, add_tag in zip(context_tokens, add_tags):
            if add_tag != -1:
                context_tokens_tmp.append(token)
        context_tokens = context_tokens_tmp
    # 合并原始文本的tokens成string，并删除空格
    context = ''.join(context_tokens).replace(' ', '')
    # 计算token在文本中的位置偏置（删除了所有空格后的文本）
    offsets = []
    cur_offset = 0
    for token in context_tokens:
        offsets.append((cur_offset, cur_offset + len(token.replace(' ', ''))))
        cur_offset = offsets[-1][1]
    match_flag = False
    labels = []
    span_offsets = []
    for tmp in re.finditer(target_string, context):
        span_offsets.append(tmp.span())
    # 如果未匹配到，则返回的labels为空队列。若匹配到，则生成labels。
    if len(span_offsets) > 0:
        match_flag = True
        for offset in offsets:
            label = 0
            for span_offset in span_offsets:
                # todo 添加违规span范围的assert
                if span_offset[0] <= offset[0] and offset[1] <= span_offset[1]:
                    label = 1
                    continue
            labels.append(label)
        if add_tags is not None:
            labels_tmp = []
            for add_tag in add_tags:
                if -1 == add_tag:
                    labels_tmp.append(0)
                else:
                    labels_tmp.append(labels.pop(0))
            labels = labels_tmp
    return match_flag, labels


# 标注一条QA样本
def label_single_qa_sample(sample, qa, recipe):
    ingredients = recipe['ingredient_dfs']
    new_directions = recipe['new_direction_dfs']
    directions = recipe['direction_dfs']

    question = qa['question']
    answer = qa['answer']
    qa_type = qa['type']

    # 走规则模型
    if qa_type in qa_type_rule:
        match_info = 'rule'
        sample['match_info'] = match_info
        # sample['ingredients'] = ingredients
        # sample['directions'] = directions
        return sample
    elif qa_type in [tp for types in qa_type_model.values() for tp in types]:
        match_info = 'cannot_match'

        # 对于无答案情况（包括已标注的‘找不到答案：N/A’和test集的无标注的‘无答案空字段：None’），进行单独处理，并直接返回结果
        if ('N/A' == answer) or (answer is None):
            match_info = 'no_answer' if answer is not None else 'test_dataset'
            tokens = [token for df in new_directions for token in df['form'].tolist()]
            sample['tokens'] = tokens
            sample['upos'] = [upos for df in new_directions for upos in df['upos_id'].tolist()]
            sample['entity'] = [entity for df in new_directions for entity in df['entity_id'].tolist()]
            sample['label'] = [0 for _ in tokens] if answer is not None else None
            sample['match_info'] = match_info
            return sample

        # 获取问题和答案的关键词，并转化为词源
        q_stopwords = ['', 'the', 'with', 'in', 'to', 'on', 'from', 'a', 'then']
        q_kws = get_keywords([qa['key_str_q']], seps=[' and ', ' '], stopwords=q_stopwords, puncts=['.', ',', ';'])
        q_kws = [(kw, lemmer.lemmatize(kw, 'v'), lemmer.lemmatize(kw, 'n')) for kw in q_kws]
        a_kws = get_keywords([qa['key_str_a']], seps=[' and ', ' '], stopwords=['', 'the'], puncts=['.', ',', ';'])
        a_kws = [(kw, lemmer.lemmatize(kw, 'v'), lemmer.lemmatize(kw, 'n')) for kw in a_kws]
        a_kws_flat = [y for x in a_kws for y in x]

        # 遍历操作步骤的文本，先判断【问题】是否匹配，再判断【答案】是否匹配
        sent_label_flag = [None for _ in new_directions]
        for idx, direction in enumerate(new_directions):
            direction_tokens = direction['form'].tolist()
            direction_tokens_lemma = direction['lemma'].tolist()
            # 判断当前句是否匹配到【问题】
            match_q_cnt = 0
            for cur_a_kws in q_kws:
                if token_match(cur_a_kws, direction_tokens + direction_tokens_lemma):
                    match_q_cnt += 1
            q_match_flag = len(q_kws) == match_q_cnt
            # 判断当前句是否匹配到【答案】
            if qa_type in qa_type_model[1]:
                # 判断单句是否匹配
                match_a_cnt = 0
                for cur_a_kws in a_kws:
                    if token_match(cur_a_kws, direction_tokens + direction_tokens_lemma):
                        match_a_cnt += 1
                a_match_flag = len(a_kws) == match_a_cnt
                if a_match_flag is True:
                    a_match_info = 'token_oneSent'
                # 若单句不匹配，则尝试双句匹配
                if (a_match_flag is False) and (idx < len(new_directions) - 1):
                    direction_tokens_2 = new_directions[idx + 1]['form'].tolist()
                    direction_tokens_lemma_2 = new_directions[idx + 1]['lemma'].tolist()
                    match_a_cnt = 0
                    for cur_a_kws in a_kws:
                        if token_match(cur_a_kws,
                                       direction_tokens + direction_tokens_lemma + direction_tokens_2 + direction_tokens_lemma_2):
                            match_a_cnt += 1
                    a_match_flag = len(a_kws) == match_a_cnt
                    if a_match_flag is True:
                        a_match_info = 'token_twoSent'
            elif qa_type in qa_type_model[2]:
                # 判断添加了角色信息的文本是否匹配
                a_match_flag, _ = string_match(answer, direction_tokens)
                if a_match_flag is True:
                    a_match_info = 'string_add'
                # 若改进后的文本无法匹配，则判断原始文本是否匹配
                if a_match_flag is False:
                    a_match_flag, _ = string_match(answer, direction_tokens, direction['id'].tolist())
                    a_match_info = 'string_ori' if a_match_flag else 'None'
            else:
                a_match_flag = None
                raise ValueError('出现了意料之外的rule qa_type')

            if q_match_flag and a_match_flag:
                sent_label_flag[idx] = 1
                if 'token_twoSent' == a_match_info:
                    sent_label_flag[idx + 1] = 1
                match_info = a_match_info
                # match_info = 'full_rule'
            else:
                sent_label_flag[idx] = 0

        # # 读取菜谱文本的token，并标注label
        # tokens = [token for df in ingredients for token in df['form'].tolist()]
        # labels = [0 for _ in tokens]
        tokens = []
        uposs = []
        entitys = []
        labels = []
        # 添加操作步骤的tokens和labels
        for idx, direction in enumerate(new_directions):
            direction_tokens = direction['form'].tolist()
            direction_tokens_lemma = direction['lemma'].tolist()
            # 添加当前步骤的token，upos，entity
            tokens.extend(direction_tokens)
            uposs.extend(direction['upos_id'].tolist())
            entitys.extend(direction['entity_id'].tolist())
            # 添加当前步骤的label
            if 1 == sent_label_flag[idx]:
                # todo 当答案匹配到多个位置时，根据问题的位置，找更近的答案
                # todo 添加duplicate说明
                if qa_type in qa_type_model[1]:
                    sent_labels = [1 if token_match(tokens, a_kws_flat) else 0 for tokens in
                                   zip(direction_tokens, direction_tokens_lemma)]
                    labels.extend(sent_labels)
                elif qa_type in qa_type_model[2]:
                    # _, sent_labels = string_match(answer, direction_tokens)
                    # labels.extend(sent_labels)
                    a_match_flag, sent_labels = string_match(answer, direction_tokens)
                    if a_match_flag is False:
                        a_match_flag, sent_labels = string_match(answer, direction_tokens, direction['id'].tolist())
                    labels.extend(sent_labels)
                else:
                    raise ValueError('出现了意料之外的rule qa_type')
            elif 0 == sent_label_flag[idx]:
                labels.extend([0 for _ in direction_tokens])

        assert len(tokens) == len(labels)
        sample['tokens'] = tokens
        sample['upos'] = uposs
        sample['entity'] = entitys
        sample['label'] = labels
        sample['match_info'] = match_info
        return sample
    else:
        print(qa_type)
        raise ValueError('出现了意料之外的qa_type')


# 自动标注
def auto_label(recipe):
    # todo 判断对于同一个菜谱，是否存在同样问句
    # assert len(recipe['qa_df']) == len(recipe['qa_df']['question'].drop_duplicates())

    samples = []
    for idx, qa in recipe['qa_df'].iterrows():
        sample = {
            'id': "{}###{}-{}###{}".format(recipe['newdoc_id'], qa['family_id'], qa['seq_id'], qa['question']),
            'question': qa['question'],
            'answer': qa['answer'],
            'context': None,
            'tokens': None,
            'upos': None,
            'entity': None,
            'offset_maping': None,
            'label': None,
            'family_id': qa['family_id'],
            'qa_type': qa['type'],
            'match_info': None,
            # 'ingredients': None,
            # 'directions': None,
        }
        # 获取样本（模型）的5个字段信息：tokens，upos，entity，label，match_info
        # 获取样本（规则）的1个字段信息：match_info
        sample = label_single_qa_sample(sample, qa, recipe)

        # 对于模型样本，拼接context，计算offset_maping
        if 'rule' != sample['match_info']:
            if sample['tokens'] is not None:
                tks = sample['tokens']
                cur_offset = -1
                offset_maping = []
                for tk in tks:
                    offset_maping.append((cur_offset + 1, cur_offset + 1 + len(tk)))
                    cur_offset = cur_offset + 1 + len(tk)
                sample['context'] = ' '.join(tks)
                sample['offset_maping'] = offset_maping

        samples.append(sample)
    return samples


# 菜谱信息分析
def analyze_recipe(recipes, mode):
    if mode is False:
        return

    ingredient_all = pd.concat([df for recipe in recipes.values() for df in recipe['ingredient_dfs']])
    direction_all = pd.concat([df for recipe in recipes.values() for df in recipe['direction_dfs']])
    new_direction_all = pd.concat([df for recipe in recipes.values() for df in recipe['new_direction_dfs']])

    # useful_cols: upos
    for col in ['entity', 'part1', 'part2', 'hidden', 'coref', 'predicate']:
        print("======{}======".format(col))
        print(ingredient_all[ingredient_all[col] != '_'][ingredient_all.columns[:10]])
    for col in ['arg{}'.format(str(i)) for i in range(1, 11)]:
        print("======{}======".format(col))
        print(ingredient_all[ingredient_all[col] != '_'][col].value_counts())

    # useful_cols: upos, entity, hidden, coref
    for col in ['part1', 'part2', 'predicate']:
        print("======{}======".format(col))
        print(direction_all[direction_all[col] != '_'][['form', 'part1', 'part2', 'coref', 'predicate']])
    for col in ['arg{}'.format(str(i)) for i in range(1, 11)]:
        print("======{}======".format(col))
        print(direction_all[direction_all[col] != '_'][col].value_counts())
    # entity: EVENT, EXPLICITINGREDIENT, IMPLICITINGREDIENT, HABITAT, TOOL
    a = direction_all[direction_all['part1'] != '_']
    # hidden: Drop, Habitat, Tool, Result, Shadow
    hid = [dep for deps in direction_all[direction_all['hidden'] != '_']['hidden'].to_list() for dep in deps.split('|')]
    b = pd.DataFrame([(h.split('=')[0], len(h.split('=')[1].split(':')), h.split('=')[1].split('.')[0]) for h in hid],
                     columns=['hid', 'cnt', 'first'])
    for val in b['hid'].unique().tolist():
        print("==={}===".format(val))
        print(b[b['hid'] == val]['cnt'].value_counts())
    pd.Series([d.split('=')[0] for d in hid]).value_counts()
    c = direction_all[['form', 'coref']]
    d = direction_all[direction_all['predicate'] != '_'][['form', 'predicate']]
    direction_all[direction_all['upos'] == 'PUNCT']['form'].value_counts(normalize=True)


# QA样本信息分析
def analyze_qa(qa_data_df, recipes, mode):
    def qa_case_analyze(recipe_id, question, recipes):
        direction_all = pd.concat([df for df in recipes[recipe_id]['direction_dfs']])

    qa_ori_all = pd.concat([recipe['qa_df'] for recipe in recipes.values()])
    # qa_ori_case = qa_ori_all[qa_ori_all['question'].apply(lambda x: x.startswith('How do you '))]
    # qa_ori_case = qa_ori_case[qa_ori_case['family_id'] == 2]
    # qa_ori_case = qa_ori_case[qa_ori_case['answer'] != 'by hand']
    # qa_ori_case['answer_prefix'] = qa_ori_case['answer'].apply(lambda x: tuple(x.split(' ')[:3]))

    if mode is False:
        return

    # qa_data_df.sort_values(['type'])[['type', 'question', 'answer']].to_csv(os.path.join(data_dir, 'qa.txt'),
    #                                                                         sep='\x01', index=None)

    # case_df = qa_data_df[(qa_data_df['type'] == 'act_ref_igdt') & (qa_data_df['match_info'] == 'cannot_match')]
    # for idx in range(len(case_df)):
    #     row = case_df.iloc[idx]
    #     recipe_id = '-'.join(row['id'].split('-')[:2])
    #     question = row['question']
    #     answer = row['answer']
    #     recipe = recipes[recipe_id]
    #     direction_all = pd.concat([df for df in recipe['direction_dfs']])
    #     pass

    # qa_all = pd.concat([recipe['qa_df'] for recipe in recipes.values()])
    # for answer in qa_all['answer'].to_list():
    #     if 'N/A' == answer:
    #         continue
    #     if re.findall('[A-Z]', answer):
    #         print(answer)

    # type = 'act_ref_place'
    # case = qa_all[qa_all['type'] == type]
    # case['the'] = case['key_str_q'].apply(lambda x: x.split(' ')[1])
    # case = case[case['the'] != 'the']
    # case2 = qa_data_df[qa_data_df['type'] == type][['id', 'question', 'answer', 'match_info']]

    # case['key_str_q'].apply(
    #     lambda x: get_keywords([x], seps=[' and ', ' '], stopwords=['', 'the', 'with'], puncts=['.', ',', ';']))

    # qa_data_df['len'] = qa_data_df['tokens'].apply(len)

    # for qa_type, tmp_df in qa_data_df.groupby(['type']):
    #     print("==={}===".format(qa_type))
    #     print(tmp_df['match_info'].value_counts(dropna=False))
    # print(qa_data_df['match_info'].value_counts(normalize=True).sort_index(ascending=False) * 100)
    # print(qa_data_df['type'].value_counts(normalize=True) * 100)
    # print(qa_data_df[['family_id', 'type']].value_counts(normalize=True).sort_index() * 100)
    # print(qa_data_df[['type', 'family_id']].value_counts(normalize=True).sort_index() * 100)

    # case = qa_data_df
    # case = case[case['type'] == 'act_ref_igdt']
    # case['question'].apply(lambda x: x[15:].split(' ')[:2][1:]).value_counts()
    #
    # case = qa_data_df
    # case = case[case['type'] == 'act_ref_place']
    # case = case[case['match_info'] == 'cannot_match']
    # idx = 222
    # recipe_id = '-'.join(case['id'].loc[idx].split('-')[:2])
    # question = case['question'].loc[idx]
    # recipe = recipes[recipe_id]
    # direction = pd.concat([df for df in recipe['direction_dfs']])

    pass


def data_process(dataset_name):
    data_train_dir = os.path.join(data_dir, 'train')
    data_vali_dir = os.path.join(data_dir, 'val')
    data_test_dir = os.path.join(data_dir, 'test')
    fields = ['id', 'form', 'lemma', 'upos', 'entity', 'part1', 'part2', 'hidden', 'coref', 'predicate',
              'arg1', 'arg2', 'arg3', 'arg4', 'arg5', 'arg6', 'arg7', 'arg8', 'arg9', 'arg10']
    # fields = None

    # for file in os.listdir(data_val_dir):
    #     print(file)
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
    recipes = []
    for d in data:
        recipe = parse_recipe(d)
        recipes.append(recipe)
    recipes = {recipe['newdoc_id']: recipe for recipe in recipes}

    # 自动标注
    all_samples = []
    for recipe_id, recipe in recipes.items():
        single_recipe_qa_samples = auto_label(recipe)
        all_samples += single_recipe_qa_samples
    data_df = pd.DataFrame(all_samples)

    # 最终的返回数据集（模型）
    data_model_df = data_df['rule' != data_df['match_info']]
    dataset_model = {
        'question': data_model_df['question'].to_list(),
        'context': data_model_df['context'].to_list(),
        'upos': data_model_df['upos'].to_list(),
        'entity': data_model_df['entity'].to_list(),
        'offset_maping': data_model_df['offset_maping'].to_list(),
        'label': data_model_df['label'].to_list(),
        'id': data_model_df['id'].to_list(),
        'answer': data_model_df['answer'].to_list(),
        'qa_type': data_model_df['qa_type'].to_list(),
    }

    # 最终的返回数据集（规则）
    data_rule_df = data_df['rule' == data_df['match_info']]
    dataset_rule = {
        'qa_data': data_rule_df[['id', 'question', 'answer', 'qa_type']],
        'recipe_data': recipes,
    }

    # 分析材料清单和操作步骤的额外信息
    analyze_recipe(recipes, False)

    # 分析自动标注数据
    analyze_qa(data_model_df, recipes, True)

    if False:
        qa_all = pd.concat([r['qa_df'] for r in recipes]).sort_values(['family_id', 'seq_id']).reset_index(drop=True)
        # inter_parse_qa_test(qa_all)
        qa_all.to_csv(os.path.join(data_dir, 'qa_val.csv'), index=None, sep='\t', encoding='utf-8')
    return dataset_model, dataset_rule


if __name__ == "__main__":
    dataset_model, dataset_rule = data_process('vali')
    print('END')
