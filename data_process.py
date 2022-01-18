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
from init_config import src_dir, data_dir

lemmer = ns.WordNetLemmatizer()

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
    assert len(q_list) == len(a_list)
    for key, value in q_list + a_list:
        recipe[0].metadata.pop(key)
    qa_list = []
    for (q_key, q_value), (a_key, a_value) in zip(q_list, a_list):
        q_id = q_key.split(' ')[1]
        a_id = a_key.split(' ')[1]
        assert q_id == a_id
        family_id, seq_id = q_id.split('-')
        qa_list.append((family_id, seq_id, q_value, a_value))
    qa_df = pd.DataFrame(qa_list, columns=['family_id', 'seq_id', 'question', 'answer'])
    qa_df['family_id'] = qa_df['family_id'].astype(int)
    qa_df['seq_id'] = qa_df['seq_id'].astype(int)
    qa_df = qa_df.sort_values(['family_id', 'seq_id']).reset_index(drop=True)
    qa_df['answer'] = qa_df['answer'].apply(lambda x: x.lstrip())
    qa_df[['question', 'answer', 'method', 'type', 'key_str_q', 'key_str_a', 'keyword_a']] \
        = qa_df.apply(parse_qa, axis=1, result_type="expand")

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
    ingredient_dfs = [pd.DataFrame([x for x in ingredient]) for ingredient in ingredients]
    direction_dfs = [pd.DataFrame([x for x in direction]) for direction in directions]
    # 根据隐藏角色信息，重写操作步骤文本
    new_direction_dfs = expand_hidden_role(direction_dfs, ingredient_dfs)
    for tmp_dfs in [ingredient_dfs, direction_dfs, new_direction_dfs]:
        for tmp_df in tmp_dfs:
            tmp_df['form'] = tmp_df['form'].apply(lambda x: x.strip().lower())
            tmp_df['lemma'] = tmp_df['lemma'].apply(lambda x: x.strip().lower())
    # 原始菜谱的文本长度
    metadata['seq_len'] = sum([len(x) for x in ingredients]) + sum([len(x) for x in directions])
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
    'count': [0],
    'act_first': [4],
    'act_ref_igdt': [1],
    'act_ref_place': [2],
    'act_ref_tool': [2],
    'in_container': [3],
    'get_middle_igdt': [3],
    'act_extent': [5],
    'full_act': [5, 6, 8, 10, 14],
    'act_duration': [7],
    'add_igdt_place': [8, 12],
    'igdt_act_ref_place': [6, 8, 12, 13],
    'igdt_amount': [9],
    'act_reason': [11, 15],
    'act_couple_igdt': [8, 12, 13],
    'act_from_where': [16],
    'place_before_act': [17],
    'what_do_you': [2, 5, 8, 9, 10],
    'how_would_you': [9],
}
type_q_regex_pattern_dict = {
    'count': ['How many actions does it take to process the (?P<keyword>.+)\?',
              'How many times is the (?P<keyword>.+) used\?',
              'How many (?P<keyword>.+) are used\?'],
    'act_first': ['(?P<keyword>.+), which comes first\?'],
    'act_ref_igdt': ['What should be (?P<keyword>.+)\?'],
    'act_ref_place': ['Where should you (?P<keyword>.+)\?'],
    'act_ref_tool': ['How do you (?P<keyword>.+)\?'],
    'in_container': ['What\'s in the (?P<keyword>.+)\?'],
    'get_middle_igdt': ['How did you get the (?P<keyword>.+)\?'],
    'act_extent': ['To what extent do you (?P<keyword>.+)\?'],
    'full_act': ['How do you (?P<keyword>.+)\?'],
    'act_duration': ['For how long do you (?P<keyword>.+)\?'],
    'add_igdt_place': ['Where do you add (?P<keyword>.+)\?'],
    'igdt_act_ref_place': ['Where do you (?P<keyword>.+)\?'],
    'igdt_amount': ['By how much do you (?P<keyword>.+)\?'],
    'act_reason': ['Why do you (?P<keyword>.+)\?'],
    'act_couple_igdt': ['What do you (?P<keyword>.+) with\?'],
    'act_from_where': ['From where do you (?P<keyword>.+)\?'],
    'place_before_act': ['Where was the (?P<keyword>.+) before (?P<keyword2>.+)\?'],
    'what_do_you': ['What do you (?P<keyword>.+)\?'],
    'how_would_you': ['How would you (?P<keyword>.+)\?'],
}
type_a_regex_pattern_dict = {
    'count': ['(?P<keyword>.+)'],
    'act_first': ['(?P<keyword>.+)'],
    'act_ref_igdt': ['(?:the )?(?P<keyword>.+)'],
    'act_ref_place': ['(?P<keyword>.+)'],
    'act_ref_tool': ['by (?P<keyword>hand)', 'by using a (?P<keyword>.+)'],
    'in_container': ['(?:the )?(?P<keyword>.+)'],
    'get_middle_igdt': ['by (?P<keyword>.+)'],
    'act_extent': ['(?:until )(?P<keyword>.+)',
                   '(?:till )?(?P<keyword>.+)'],
    'full_act': ['(?P<keyword>.+)'],
    'act_duration': ['(?:for )?(?P<keyword>.+)'],
    'add_igdt_place': ['(?:to )?(?P<keyword>.+)'],
    'igdt_act_ref_place': ['(?P<keyword>.+)'],
    'igdt_amount': ['(?P<keyword>.+)'],
    'act_reason': ['(?:so )(?P<keyword>.+)',
                   '(?:to )(?P<keyword>.+)',
                   '(?:for )(?P<keyword>.+)',
                   '(?P<keyword>.+)'],
    'act_couple_igdt': ['(?:with )?(?P<keyword>.+)'],
    'act_from_where': ['(?:from )?(?P<keyword>.+)'],
    'place_before_act': ['(?P<keyword>.+)'],
    'what_do_you': ['(?P<keyword>.+)'],
    'how_would_you': ['(?P<keyword>.+)'],
}
type_sep_dict = {
    'count': [],
    'act_first': [],
    'act_ref_igdt': [' and ', ','],
    'act_ref_place': [' and ', ','],
    'act_ref_tool': [' and ', ','],
    'in_container': [' and ', ','],
    'get_middle_igdt': [' and '],
    'act_extent': [' and '],
    'full_act': [],
    'act_duration': [],
    'add_igdt_place': [],
    'igdt_act_ref_place': [],
    'igdt_amount': [],
    'act_reason': [],
    'act_couple_igdt': [' and ', ','],
    'act_from_where': [],
    'place_before_act': [],
    'what_do_you': [],
    'how_would_you': [],
}


# 解析QA对
# qa_all[['question', 'answer', 'method', 'type', 'key_str_q', 'key_str_a', 'keyword_a']] = qa_all.apply(parse_qa, axis=1, result_type="expand")
def parse_qa(qa_row):
    question = qa_row['question'].replace(r'\"', "").lstrip()
    answer = qa_row['answer'].replace(r'\"', "").lstrip()
    family_id = qa_row['family_id']

    # method:采用什么方式回答问题。cat_0和cat_4的问题通过规则解答，其余类别的问题通过模型解答。
    # type:问答类别标签
    # key_str_q:问题核心文本
    # key_str_a:答案核心文本
    # keyword_a:答案关键字

    # cat_0：3类 统计操作次数，工具使用次数，食材使用个数
    # cat_4：1类 两个操作里，那个操作更早进行？
    # cat_1：1类 操作所涉及的食材是什么？
    # cat_2：2类 操作所涉及的场所是什么？操作所涉及的工具是什么？
    # cat_3：2类 某个容器里有什么？如何获得中间食材？
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
                # 判断answer是不是NA
                if 'N/A' == answer:
                    assert family_id == 18
                    method = 'no_answer'
                    type = 'no_answer'
                    key_str_q = regex_q.group(1)
                    key_str_a = None
                    keyword_a = None
                    return question, answer, method, type, key_str_q, key_str_a, keyword_a
                # 遍历当前qa类型下的所有answer正则模板
                for a_regex_pattern in type_a_regex_pattern_dict[qa_type]:
                    regex_a = re.match(a_regex_pattern, answer)
                    # 匹配到answer的正则模板
                    if regex_a:
                        if qa_type in ['count', 'act_first']:
                            method = 'rule'
                        else:
                            method = 'model'
                        type = qa_type
                        assert regex_q.group(0) == question
                        assert family_id in type_family_id_dict[qa_type]
                        key_str_q = regex_q.group(1)
                        key_str_a = regex_a.group(1)
                        keyword_a = [key_str_a]
                        for sep in type_sep_dict[qa_type]:
                            keyword_a = [y.strip() for x in keyword_a for y in x.split(sep)]
                        return question, answer, method, type, key_str_q, key_str_a, keyword_a
    # assert family_id > 17
    print(qa_row, flush=True)
    raise ValueError('出现了意料之外的QA模式')


def get_keywords(str_list, seps, stopwords=[], puncts=[]):
    keywords = str_list
    for punct in puncts:
        keywords = [kw.replace(punct, '') for kw in keywords]

    for sep in seps:
        keywords = [y.strip() for x in keywords for y in x.split(sep)]
        keywords = [kw for kw in keywords if kw not in stopwords]

    return keywords


def inter_parse_qa_test(qa_df):
    for idx, qa_row in qa_df.iterrows():
        if qa_row['question'].__contains__("How do you shell the peas in the bowl?"):
            print(qa_row['question'])
        print('{}\t{}'.format(qa_row['question'], qa_row['answer']), flush=True)
        tmp = parse_qa(qa_row)


# 根据隐藏角色信息，重写文本。
def expand_hidden_role(directions, ingredients):
    def join_role_items(role_items, upos_map, entity):
        role_items_plus = []
        # 标记upos和entity属性
        for role_item in role_items:
            role_item = [[token, upos_map.get(token, 'NOUN'), 'I-' + entity] for token in role_item]
            role_item[0][2] = 'B-' + entity
            role_items_plus.append(role_item)
        # 添加连接词
        if 1 == len(role_items_plus):
            new_role_items = role_items_plus[0]
        else:
            new_role_items = []
            for idx in range(len(role_items_plus) - 1):
                new_role_items += role_items_plus[idx]
                new_role_items.append([',', 'PUNCT', 'O'])
            new_role_items[-1] = ['and', 'CCONJ', 'O']
            new_role_items += role_items_plus[idx + 1]
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
            if '_' == row['hidden']:
                direction_new.append(pd.DataFrame([row.to_dict()], columns=direction.columns))
            else:
                cur_token = [pd.DataFrame([row.to_dict()], columns=direction.columns)]
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
                        add_tokens = [['to', 'PART', 'O'], ['get', 'VERB', 'O']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O']]
                        if check_last_punct(direction_new) is False:
                            add_tokens = [[',', 'PUNCT', 'O']] + add_tokens
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
                        add_tokens = [['in', 'ADP', 'O']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O']]
                        if check_last_punct(direction_new + cur_token) is False:
                            add_tokens = [[',', 'PUNCT', 'O']] + add_tokens
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
                        if add_tokens[0][0] in ['hand', 'hands']:
                            add_tokens[0][0] = 'hand'
                            add_tokens = [['by', 'ADP', 'O']] \
                                         + add_tokens \
                                         + [[',', 'PUNCT', 'O']]
                        else:
                            add_tokens = [['by', 'ADP', 'O'], ['using', 'VERB', 'O'], ['a', 'DET', 'O']] \
                                         + add_tokens \
                                         + [[',', 'PUNCT', 'O']]
                        if check_last_punct(direction_new + cur_token) is False:
                            add_tokens = [[',', 'PUNCT', 'O']] + add_tokens
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
                        add_tokens = [['the', 'DET', 'O']] \
                                     + add_tokens \
                                     + [[',', 'PUNCT', 'O']]
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


# 标注一条QA样本
def label_single_qa_sample(sample, qa, recipe):
    def token_match(tokens, token_list):
        for token in tokens:
            if token in token_list:
                return True
        return False

    ingredients = recipe['ingredient_dfs']
    new_directions = recipe['new_direction_dfs']
    directions = recipe['direction_dfs']

    question = qa['question']
    answer = qa['answer']
    qa_type = qa['type']

    rule_match_type = {
        1: ['act_ref_place', 'act_ref_tool', 'act_ref_igdt', 'full_act'],
        2: ['igdt_act_ref_place']
    }
    # 走规则模型
    if qa_type in ['count', 'act_first']:
        # todo 不标注
        return None
    elif qa_type in ['act_ref_place', 'act_ref_tool', 'act_ref_igdt', 'full_act']:
        if 'igdt_act_ref_place' == qa_type:
            print('')
            pass
        q_stopwords = ['', 'the', 'with', 'in', 'to', 'on', 'from', 'a']
        del_puncts = ['.', ',', ';']
        match_info = 'cannot_match'
        # 清洗输入文本1：读取操作步骤的原始token和词源token
        drts_tokens = [direction['form'].tolist() for direction in new_directions]
        drts_tokens_lemma = [direction['lemma'].tolist() for direction in new_directions]
        # 清洗输入文本2：剔除标点符号，部分文本未能够将标点符号分词为独立token
        for del_p in del_puncts:
            drts_tokens = [[t.replace(del_p, '') if len(t) > 1 else t for t in d] for d in drts_tokens]
            drts_tokens_lemma = [[t.replace(del_p, '') if len(t) > 1 else t for t in d] for d in drts_tokens_lemma]
        # 获取问题和答案的关键词，并转化为词源
        q_kws = get_keywords([qa['key_str_q']], seps=[' and ', ' '], stopwords=q_stopwords, puncts=['.', ',', ';'])
        q_kws = [(kw, lemmer.lemmatize(kw, 'v'), lemmer.lemmatize(kw, 'n')) for kw in q_kws]
        a_kws = get_keywords([qa['key_str_a']], seps=[' and ', ' '], stopwords=['', 'the'], puncts=['.', ',', ';'])
        a_kws = [(kw, lemmer.lemmatize(kw, 'v'), lemmer.lemmatize(kw, 'n')) for kw in a_kws]
        a_kws_flat = [y for x in a_kws for y in x]

        # for idx, new_direction in enumerate(new_directions):
        #     # 读取操作步骤的原始token和词源token
        #     direction_tokens = [t for t in new_direction['form'].tolist()]
        #     direction_tokens_lemma = [t for t in new_direction['lemma'].tolist()]
        #     # 剔除标点符号，部分文本未能够将标点符号分词为独立token
        #     for del_p in del_puncts:
        #         direction_tokens = [t.replace(del_p, '') if len(t) > 1 else t for t in direction_tokens]
        #         direction_tokens_lemma = [t.replace(del_p, '') if len(t) > 1 else t for t in direction_tokens_lemma]

        # 遍历操作步骤的文本，判断问题是否匹配，判断答案是否匹配
        sent_label_flag = [None for _ in new_directions]
        for idx, (direction_tokens, direction_tokens_lemma) in enumerate(zip(drts_tokens, drts_tokens_lemma)):
            # 判断当前句是否匹配到问题
            match_q_cnt = 0
            for kw, kw_v_lemma, kw_n_lemma in q_kws:
                if token_match((kw, kw_v_lemma, kw_n_lemma), direction_tokens + direction_tokens_lemma):
                    match_q_cnt += 1
            if len(q_kws) == match_q_cnt:
                q_match_flag = True
            else:
                q_match_flag = False
            # 判断当前句是否匹配到答案
            if qa_type in rule_match_type[1]:
                match_a_cnt = 0
                for kw, kw_v_lemma, kw_n_lemma in a_kws:
                    if token_match((kw, kw_v_lemma, kw_n_lemma), direction_tokens + direction_tokens_lemma):
                        match_a_cnt += 1
                if len(a_kws) == match_a_cnt:
                    a_match_flag = True
                else:
                    a_match_flag = False
            elif qa_type in rule_match_type[1]:
                a_match_flag = None
            else:
                a_match_flag = None
                raise ValueError('出现了意料之外的rule type')

            if q_match_flag and a_match_flag:
                sent_label_flag[idx] = 1
                match_info = 'full_rule'
            else:
                sent_label_flag[idx] = 0

        # 读取菜谱文本的token，并标注label
        tokens = [token for df in ingredients for token in df['form'].tolist()]
        labels = [0 for _ in tokens]
        # 添加操作步骤的tokens和labels
        for idx, (direction_tokens, direction_tokens_lemma) in enumerate(zip(drts_tokens, drts_tokens_lemma)):
            tokens.extend(direction_tokens)
            if 1 == sent_label_flag[idx]:
                # todo 当答案匹配到多个位置时，根据问题的位置，找更近的答案
                # todo 添加duplicate说明
                labels.extend([1 if token_match(tokens, a_kws_flat) else 0
                               for tokens in zip(direction_tokens, direction_tokens_lemma)])
            elif 0 == sent_label_flag[idx]:
                labels.extend([0 for _ in direction_tokens_lemma])
        sample['tokens'] = tokens
        sample['label'] = labels
        sample['match_info'] = match_info
        return sample

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

    # 没匹配到答案
    match_info = 'cannot_match'
    sample['label'] = [0 for _ in tokens]
    sample['match_info'] = match_info
    return sample


# 自动标注
def auto_label(recipe):
    # todo 判断对于同一个菜谱，是否存在同样问句
    # assert len(recipe['qa_df']) == len(recipe['qa_df']['question'].drop_duplicates())

    samples = []
    for idx, qa in recipe['qa_df'].iterrows():
        sample = {
            'id': "{}-{}".format(recipe['newdoc_id'], qa['question']),
            'question': qa['question'],
            'answer': qa['answer'],
            'context': None,
            'tokens': None,
            'label': None,
            'offset_maping': None,
            'family_id': qa['family_id'],
            'type': qa['type'],
            'match_info': None,
        }
        # 获取样本的3个字段信息：tokens，label，match_info
        sample = label_single_qa_sample(sample, qa, recipe)
        if sample is not None:
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

    # todo useful_cols: upos
    for col in ['entity', 'part1', 'part2', 'hidden', 'coref', 'predicate']:
        print("======{}======".format(col))
        print(ingredient_all[ingredient_all[col] != '_'][ingredient_all.columns[:10]])
    for col in ['arg{}'.format(str(i)) for i in range(1, 11)]:
        print("======{}======".format(col))
        print(ingredient_all[ingredient_all[col] != '_'][col].value_counts())

    # todo useful_cols: upos, entity, hidden, coref
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

    for qa_type, tmp_df in qa_data_df.groupby(['type']):
        print("==={}===".format(qa_type))
        print(tmp_df['match_info'].value_counts(dropna=False))

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

    # 分析材料清单和操作步骤的额外信息
    analyze_recipe(recipes, False)

    # 分析自动标注数据
    analyze_qa(data_df, recipes, True)

    dataset = {
        'question': data_df['question'].to_list(),
        'context': data_df['context'].to_list(),
        'label': data_df['label'].to_list(),
        'label_offset': data_df['offset_maping'].to_list(),
        'id': data_df['id'].to_list(),
        'answer': data_df['answer'].to_list(),
    }
    if False:
        qa_all = pd.concat([r['qa_df'] for r in recipes]).sort_values(['family_id', 'seq_id']).reset_index(drop=True)
        # inter_parse_qa_test(qa_all)
        qa_all.to_csv(os.path.join(data_dir, 'qa_val.csv'), index=None, sep='\t', encoding='utf-8')
    return dataset


if __name__ == "__main__":
    dataset = data_process('vali')
    print('END')
