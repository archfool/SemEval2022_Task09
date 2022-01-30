# -*- coding: utf-8 -*-
"""
Created on 2022/1/29 19:23
author: ruanzhihao_archfool
"""
import os
import sys
import pandas as pd
import numpy as np
import re
import nltk.stem as ns
from datasets import load_metric
import json

lm = ns.WordNetLemmatizer()

from data_process import data_process, token_match, qa_type_rule, type_q_regex_pattern_dict, get_keywords, q_stopwords


def parse_id(row):
    id = row['id']
    recipe_id, question_id, question = id.split('###')
    family_id = question_id.split('-')[0]
    return recipe_id, question_id


# 解析hidden，返回字典，元素名称里的下划线未被替换
def parse_hidden(hiddens, reserve_idx=False):
    if hiddens == '_':
        return {}

    hidden_dict = {hidden.split('=')[0].lower(): [hid for hid in hidden.split('=')[1].split(':')]
                   for hidden in hiddens.split('|')}
    if reserve_idx is False:
        hidden_dict = {hid_name: [hid_value.split('.')[0] for hid_value in hid_values]
                       for hid_name, hid_values in hidden_dict.items()}
    return hidden_dict


# 解析coref，返回列表，元素名称里的下划线未被替换
def parse_coref(corefs, reserve_idx=False):
    if corefs == '_':
        return []
    coref_list = corefs.split(':')
    if reserve_idx == False:
        coref_list = [coref.split('.')[0] for coref in coref_list]
    return coref_list


def collect_segment_items(key_verb_row, segment, argx_col):
    get_segment_entity_info()
    get_segment_argx_info(segment, argx_col)
    parse_hidden(key_verb_row['hidden'])
    parse_coref(key_verb_row['coref'])


# 提取hidden列和coref列的标注信息，并分割为token，格式为list[list[]] todo 带验证
def collect_annotation_items(segment, reserve_idx=False):
    items = []
    for hidden in segment['hidden'].tolist():
        hidden_dict = parse_hidden(hidden)
        hidden_items = [item.split('_') for items in hidden_dict.values() for item in items]
        items.extend(hidden_items)
    for coref in segment['coref'].tolist():
        coref_list = parse_coref(coref)
        coref_items = [item.split('_') for item in coref_list]
        items.extend(coref_items)
    return items


# 提取hidden列中，命中target的所有元素
def collect_hidden(hiddens, target):
    lemme_target = '_'.join([lm.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    collected_items = []
    for values in hiddens.tolist():
        value_list = [y for x in values.split('|') for y in x.split('=')[1].split(':')] if values != '_' else []
        for value in value_list:
            lemme_value = '_'.join([lm.lemmatize(token, 'n') for token in value.split('.')[0].split('_')])
            if lemme_value == lemme_target:
                collected_items.append(value)
    return collected_items


# 提取coref列中，命中target的所有元素
def collect_coref(corefs, target):
    lemme_target = '_'.join([lm.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    collected_items = []
    for values in corefs.tolist():
        for value in values.split(':'):
            lemme_value = '_'.join([lm.lemmatize(token, 'n') for token in value.split('.')[0].split('_')])
            if lemme_value == lemme_target:
                collected_items.append(value)
    return collected_items


# 提取菜单文本段落的entity字段的信息
def get_segment_entity_info(direction_segment, anchor_verb_idx=None):
    segs = {}
    segs['habitat'] = direction_segment[
        (direction_segment['entity'] == 'B-HABITAT') | (direction_segment['entity'] == 'I-HABITAT')]
    segs['tool'] = direction_segment[
        (direction_segment['entity'] == 'B-TOOL') | (direction_segment['entity'] == 'I-TOOL')]
    segs['igdt'] = direction_segment[
        (direction_segment['entity'] == 'B-EXPLICITINGREDIENT')
        | (direction_segment['entity'] == 'I-EXPLICITINGREDIENT')
        | (direction_segment['entity'] == 'B-IMPLICITINGREDIENT')
        | (direction_segment['entity'] == 'I-IMPLICITINGREDIENT')]

    # 添加食材、容器、工具信息
    seg_infos = {key: [] for key in segs.keys()}
    for type, seg in segs.items():
        info = []
        for _, row in seg.iterrows():
            if row['entity'].startswith('B-') and len(info) > 0:
                seg_infos[type].append(' '.join(info))
                info = []
            info.append(row['form'])
        if len(info) > 0:
            seg_infos[type].append(' '.join(info))

    # 添加关键动词信息
    if anchor_verb_idx is not None:
        iloc_idx = direction_segment.index.get_loc(anchor_verb_idx)
        info = []
        for _, row in direction_segment[iloc_idx:].iterrows():
            # 遇到event的head，则判断是否队列中存在event。是则跳出
            if row['entity'].startswith('B-EVENT') and len(info) > 0:
                break
            # 遇到event类型，则加入队列。否则跳出
            if row['entity'] == 'B-EVENT' or row['entity'] == 'I-EVENT':
                info.append(row['form'])
            else:
                break
        seg_infos['act'] = info
        if len(info) > 1:
            print("act longer than one word: {}".format(' '.join(info)))

    return seg_infos


# 提取菜单文本段落的argX字段的信息
def get_segment_argx_info(direction_segment, col_name):
    segs = {}
    segs['v'] = direction_segment[
        (direction_segment[col_name] == 'B-V')
        | (direction_segment[col_name] == 'I-V')
        | (direction_segment[col_name] == 'D-V')]
    segs['attribute'] = direction_segment[
        (direction_segment[col_name] == 'B-Attribute') | (direction_segment[col_name] == 'I-Attribute')]
    segs['instrument'] = direction_segment[
        (direction_segment[col_name] == 'B-Instrument') | (direction_segment[col_name] == 'I-Instrument')]
    segs['patient'] = direction_segment[
        (direction_segment[col_name] == 'B-Patient') | (direction_segment[col_name] == 'I-Patient')]

    # 添加attribute信息
    seg_infos = {key: [] for key in segs.keys()}
    for type, seg in segs.items():
        info = []
        for _, row in seg.iterrows():
            if row[col_name].startswith('B-') and len(info) > 0:
                seg_infos[type].append(' '.join(info))
                info = []
            info.append(row['form'])
        if len(info) > 0:
            seg_infos[type].append(' '.join(info))

    return seg_infos


# 判断若干个关键词keywords，是否全部被包含在sent_tokens中。每个关键词keyword包含它的多个时态
def token_states_all_in_sent(keywords, sent_tokens):
    match_cnt = 0
    for keyword in keywords:
        if token_match(keyword, sent_tokens):
            match_cnt += 1
    if len(keywords) == match_cnt:
        return True
    else:
        return False


# 根据文本，定位到相应的句子
def locate_direction(key_string, data_drt):
    q_kws = get_keywords([key_string], seps=[' and ', ' '], stopwords=q_stopwords, puncts=['.', ',', ';'])
    q_kws = [(kw, lm.lemmatize(kw, 'v'), lm.lemmatize(kw, 'n')) for kw in q_kws]

    matched_drts = []
    for seq_id, direction in data_drt.groupby('seq_id'):
        direction_tokens = direction['form'].tolist()
        direction_tokens_lemma = direction['lemma'].tolist()
        # 判断当前句是否匹配到【问题】
        match_flag = token_states_all_in_sent(q_kws, direction_tokens + direction_tokens_lemma)
        if match_flag:
            matched_drts.append([seq_id, direction])
            continue

    return matched_drts, q_kws


# 截取关键动词的关联上下文
def locate_direction_segment(idx, direction):
    # 若关键动词，在argX列有标识，能够提取出上下文，则返回相关上下文
    for col_name in ['arg{}'.format(str(i)) for i in range(1, 11)]:
        if direction.iloc[idx][col_name] != '_' and 'V' == direction.iloc[idx][col_name].split('-')[1]:
            seg_df = direction[direction[col_name] != '_']
            return seg_df, col_name
    # 若关键动词，在argX列没有标识，不能够提取出上下文，则返回关键动词的当前行
    seg_df = direction[idx:idx + 1]
    return seg_df, None


# 根据相应列的取值，截取操作步骤的片段 todo 待验证
def get_conditional_segment(old_segment, col_name, col_values, joint_rule='or'):
    flag = None
    for col_value in col_values:
        if joint_rule == 'or':
            flag = old_segment[col_name] == col_value if flag is None else flag | (old_segment[col_name] == col_value)
        elif joint_rule == 'and':
            flag = old_segment[col_name] == col_value if flag is None else flag & (old_segment[col_name] == col_value)
        else:
            raise Exception('Unknow joint_rule name!')
    new_segment = old_segment[flag]
    return new_segment


# 在一句操作步骤中，定位到关键词的位置
def get_keyword_loc(keyword, direction):
    keyword = [keyword, lm.lemmatize(keyword, 'v')]

    idxs = []
    for r_idx, row in direction.iterrows():
        if (keyword[0] in [row['form'], row['lemma']]) or (keyword[1] in [row['form'], row['lemma']]):
            idxs.append(r_idx)

    return idxs


# 连接item成完整的通顺句子
def join_items(item_list, empty_return=''):
    # 若输入队列为空，则返回空字符串
    if len(item_list) == 0:
        return empty_return
    # 如果队列的元素还是队列，则将队列元素转换为字符串
    if isinstance(item_list[0], list):
        item_list = [' '.join(item) for item in item_list]
    # 按照口语习惯，拼接item
    ret_string = ' and '.join([x for x in [', '.join(item_list[:-1]), item_list[-1]] if x != ''])
    # ret_string = ret_string.replace('_', ' ')
    return ret_string


if __name__ == '__main__':
    print('END')
