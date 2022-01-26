# -*- coding: utf-8 -*-
"""
Created on 2022/1/23 18:04
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import nltk.stem as ns

lemmer = ns.WordNetLemmatizer()

from init_config import src_dir, data_dir
from data_process import data_process, token_match, qa_type_rule, type_q_regex_pattern_dict, get_keywords, q_stopwords


def parse_hidden(hiddens, reserve_idx=False):
    if hiddens == '_':
        return {}

    hidden_dict = {hidden.split('=')[0]: [hid for hid in hidden.split('=')[1].split(':')]
                   for hidden in hiddens.split('|')}
    if reserve_idx is False:
        hidden_dict = {hid_name: [hid_value.split('.')[0] for hid_value in hid_values]
                       for hid_name, hid_values in hidden_dict.items()}
    return hidden_dict


def parse_id(row):
    id = row['id']
    recipe_id, question_id, question = id.split('###')
    family_id = question_id.split('-')[0]
    return recipe_id, question_id


def count_times_hidden(hiddens, target):
    count = 0
    item = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    for values in hiddens.tolist():
        if values != '_':
            value_list = [y.split('.')[0] for x in values.split('|') for y in x.split('=')[1].split(':')]
            if item in value_list:
                count += 1
    return count


def count_times_coref(corefs, target):
    count = 0
    item = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    for value in corefs.tolist():
        if value != '_':
            if item == value.split('.')[0]:
                count += 1
    return count


def count_nums_hidden(hiddens, target):
    item = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    candidate_items = []
    for values in hiddens.tolist():
        if values != '_':
            value_list = [y for x in values.split('|') for y in x.split('=')[1].split(':')]
            for value in value_list:
                if value.startswith(item):
                    candidate_items.append(value)
    return candidate_items


def count_nums_coref(corefs, target):
    item = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    candidate_items = []
    for value in corefs.tolist():
        if value != '_':
            if value.startswith(item):
                candidate_items.append(value)
    return candidate_items


def act_first(question, direction_dfs):
    directions = pd.concat(direction_dfs)
    question = question.lower().replace('(', '\(').replace(')', '\)')
    # tokens_list = [[token for token in tokens.split(' ') if token != ''] for tokens in question.split(' and ')]
    tokens_list = [tokens for tokens in question.split(' and ')]
    tokens_list = [get_keywords([ts], seps=[',', ' '], stopwords=q_stopwords, puncts=['.', ';']) for ts in tokens_list]
    tokens_list = [[(tk, lemmer.lemmatize(tk, 'v'), lemmer.lemmatize(tk, 'n')) for tk in tks] for tks in tokens_list]

    match_result = {}
    # 遍历问题的前后半句的所有组合方式
    for sep_idx in range(1, len(tokens_list)):
        # 分割问题为前半句和后半句
        first = [token for tokens in tokens_list[:sep_idx] for token in tokens]
        second = [token for tokens in tokens_list[sep_idx:] for token in tokens]
        # 判断前半句和后半句的首词，是否为动词
        verb_list = directions[directions['upos'] == 'VERB']['lemma'].tolist()
        if not (first[0][1] in verb_list and second[0][1] in verb_list):
            continue
        # 记录匹配到前半句和后半句的句子的idx
        first_sent_idxs = []
        second_sent_idxs = []
        # 遍历操作步骤的所有句子
        for sent_idx, direction in enumerate(direction_dfs):
            direction_tokens = direction['form'].tolist()
            direction_tokens_lemma = direction['lemma'].tolist()
            # 判断前半句问题，是否匹配到当前文本段落
            first_match_cnt = 0
            for f in first:
                if token_match(f, direction_tokens + direction_tokens_lemma):
                    first_match_cnt += 1
            if first_match_cnt == len(first):
                first_sent_idxs.append(sent_idx)
            # 判断后半句问题，是否匹配到当前文本段落
            second_match_cnt = 0
            for s in second:
                if token_match(s, direction_tokens + direction_tokens_lemma):
                    second_match_cnt += 1
            if second_match_cnt == len(second):
                second_sent_idxs.append(sent_idx)

        if len(first_sent_idxs) > 0 and len(second_sent_idxs) > 0:
            match_result[sep_idx] = (first_sent_idxs, second_sent_idxs)
    if len(match_result) == 0:
        ret = 'N/A'
    else:
        first_idxs = [idx for idxs, _ in match_result.values() for idx in idxs]
        second_idxs = [idx for _, idxs in match_result.values() for idx in idxs]
        if len(first_idxs) == 0 or len(second_idxs) == 0:
            ret = 'N/A'
        elif sum(first_idxs) / len(first_idxs) < sum(second_idxs) / len(second_idxs):
            ret = 'the first event'
        elif sum(first_idxs) / len(first_idxs) > sum(second_idxs) / len(second_idxs):
            ret = 'the second event'
        else:
            ret = 'equal'
    return ret


def place_before_act(igdt, act, new_direction_dfs, old_direction_dfs):
    # 1.重定位igdt名字

    # 2.找到同时包含igdt和act的句子
    # 提取关键词
    keywords = get_keywords([igdt, act], seps=[',', ' '], stopwords=q_stopwords, puncts=['.', ';'])
    keywords = [(kw, lemmer.lemmatize(kw, 'v'), lemmer.lemmatize(kw, 'n')) for kw in keywords]
    sent_idx = -1
    find_keywords_flag = False
    # 遍历所有操作步骤
    for sent_idx, new_direction in enumerate(new_direction_dfs):
        new_direction_tokens = new_direction['form'].tolist()
        new_direction_tokens_lemma = new_direction['lemma'].tolist()
        keywords_match_cnt = 0
        # 统计匹配到的关键词的数量
        for keyword in keywords:
            if token_match(keyword, new_direction_tokens + new_direction_tokens_lemma):
                keywords_match_cnt += 1
        # 判断关键词是否全被匹配到
        if keywords_match_cnt == len(keywords):
            find_keywords_flag = True
            break

    # 3.若找到关键句，则定位act的token位置
    verb_token_idx = -1
    if find_keywords_flag:
        # 提取动词关键词
        verb_keyword = get_keywords([act], seps=[',', ' '], stopwords=q_stopwords, puncts=['.', ';'])[0]
        verb_keyword = lemmer.lemmatize(verb_keyword, 'v')
        # 匹配动词关键词
        for verb_token_idx in range(len(old_direction_dfs[sent_idx]) - 1, -1, -1):
            if (verb_keyword == old_direction_dfs[sent_idx]['lemma'].iloc[verb_token_idx]) \
                    or (verb_keyword == old_direction_dfs[sent_idx]['form'].iloc[verb_token_idx]):
                break
    # 若找不到关键句，则直接返回None
    else:
        return None

    # 在定位到igdt/act的所在行后，找到相关的place
    def get_igdt_or_act_from_arg_and_entity(row, direction, focus_type, target_type):
        for col_name in ['arg{}'.format(str(i)) for i in range(1, 11)]:
            # if row[col_name] in ['B-{}'.format(focus_type), 'I-{}'.format(focus_type)]:
            if row[col_name] != '_':
                # verb_rows = direction[direction[col_name] == 'B-V']

                target_rows = direction[
                    (direction[col_name] != '_')
                    & (
                            (direction['entity'] == 'B-{}'.format(target_type))
                            | (direction['entity'] == 'I-{}'.format(target_type))
                    )
                    ]

                # 未提取到信息
                if len(target_rows) == 0:
                    continue
                # 提取place信息
                elif 'HABITAT' == target_type:
                    places = set([place.split('.')[0] for place in target_rows['coref'].tolist()])
                    place = ' '.join([token for place in places for token in place.split('_')])
                    # place = ' '.join(target_rows['form'].tolist())
                    return place
                # 提取act信息
                elif 'EVENT' == target_type:
                    target_rows = target_rows[target_rows['entity'] == 'B-EVENT']
                    if len(target_rows) == 0:
                        continue
                    else:
                        hiddens = parse_hidden(target_rows.iloc[0]['hidden'], False)
                        # 若hidden列包含Habitat字段信息，则直接取出place
                        if 'Habitat' in hiddens.keys():
                            place = hiddens['Habitat'][0].replace('_', ' ')
                            return place
                else:
                    raise ValueError("")

        return None

    # 4.往前反向搜索，找到igdt的place
    place = None
    # 获取igdt的字符串
    igdt_keywords = get_keywords([igdt], seps=[',', ' ', ' and '], stopwords=[], puncts=['.', ';'])
    igdt_keywords_str = '_'.join(igdt_keywords)
    # 遍历操作步骤的所有sents
    for igdt_sent_idx in range(sent_idx, -1, -1):
        old_direction = old_direction_dfs[igdt_sent_idx]
        start_token_idx = verb_token_idx if igdt_sent_idx == sent_idx else len(old_direction_dfs[igdt_sent_idx])
        # 遍历一条sent下的所有rows
        for igdt_token_idx in range(start_token_idx - 1, -1, -1):
            row = old_direction.iloc[igdt_token_idx]
            # 提取coref列的目标食材信息
            cur_igdt = row['coref'].split('.')[0]
            # 提取hidden的目标食材信息
            hiddens = parse_hidden(row['hidden'], reserve_idx=False)
            cur_igdt_list = hiddens.get('Drop', []) + hiddens.get('Shadow', [])

            # 当前词为名词，判断是否在coref列显式命中食材
            if igdt_keywords_str == cur_igdt:
                # 尝试从原始上下文文本中，获取显式位置信息
                habitat = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'Patient', 'HABITAT')
                # 原始文本中，显式包含位置信息
                if habitat is not None:
                    place = habitat
                # 原始文本中，不包含位置信息。则去上下文的动词处，寻找是否隐式包含place/HABITAT信息。
                else:
                    place = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'Patient', 'EVENT')
            # 当前词为动词，判断是否在hidden列隐式命中食材
            elif igdt_keywords_str in cur_igdt_list:
                # 若hidden列包含Habitat字段信息，则直接取出place
                if 'Habitat' in hiddens.keys():
                    place = hiddens['Habitat'][0].replace('_', ' ')
                # 若hidden列未包含Habitat字段信息，则找act的上下文tokens，尝试提取location/Habitat
                else:
                    place = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'V', 'HABITAT')
            # 都没命中，进入进入下一个token的过程
            else:
                place = None

            # 如果命中place，则返回
            if place is not None:
                return place


    return place


def rule_for_qa(dataset):
    qa_df = dataset['qa_data']
    recipes = dataset['recipe_data']

    def get_answer_by_rule(row, recipes=recipes):
        recipe = recipes[row['recipe_id']]
        qa_type = row['qa_type']
        question = row['question']
        answer = row['answer']
        key_context = row['context']
        direction_dfs = recipe['direction_dfs']
        new_direction_dfs = recipe['new_direction_dfs']
        directions = pd.concat(direction_dfs)

        if 'act_first' == qa_type:
            ret = act_first(key_context, new_direction_dfs)
            return ret
        elif 'place_before_act' == qa_type:
            igdt, act = key_context.split('|')
            place = place_before_act(igdt, act, new_direction_dfs, direction_dfs)
            return 'N/A' if place is None else place
        elif 'count_times' == qa_type:
            count_hidden = count_times_hidden(directions['hidden'], key_context)
            count_coref = count_times_coref(directions['coref'], key_context)
            count = count_hidden + count_coref
            count = 'N/A' if count == 0 else count
            return count
        elif 'count_nums' == qa_type:
            candidate_hidden = count_nums_hidden(directions['hidden'], key_context)
            candidate_coref = count_nums_coref(directions['coref'], key_context)
            num = len(set(candidate_hidden + candidate_coref))
            num = 'N/A' if num == 0 else num
            return num
        elif 'get_result' == qa_type:

            return ''
        elif 'result_component' == qa_type:

            return ''
        else:
            raise ValueError('invalid rule qa_type')

    qa_df[['recipe_id', 'question_id']] = qa_df.apply(parse_id, axis=1, result_type="expand")
    if True:
        qa_df['pred_answer'] = qa_df.apply(get_answer_by_rule, axis=1)
    else:
        qa_df['pred_answer'] = None
        for idx in range(len(qa_df)):
            x = get_answer_by_rule(qa_df.iloc[idx])

    return qa_df


if __name__ == '__main__':
    dataset_model_vali, dataset_rule_vali = data_process('vali')
    rule_result = rule_for_qa(dataset_rule_vali)
    print(rule_result['qa_type'].value_counts(normalize=True))
    rule_result['flag'] = rule_result.apply(lambda r: 1 if str(r['answer']) == str(r['pred_answer']) else 0, axis=1)
    for qa_type in ['act_first', 'place_before_act', 'count_times', 'count_nums']:
        print('=========={}=========='.format(qa_type))
        print(rule_result[rule_result['qa_type'] == qa_type]['flag'].value_counts(normalize=True))
    print('END')
