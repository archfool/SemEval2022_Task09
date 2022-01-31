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
def get_segment_entity_info(direction_segment, anchor_verb_idx=None, argx_col_name=None):
    segs = {}
    segs['habitat'] = direction_segment[
        (direction_segment['entity'] == 'B-HABITAT') | (direction_segment['entity'] == 'I-HABITAT')]
    segs['tool'] = direction_segment[
        (direction_segment['entity'] == 'B-TOOL') | (direction_segment['entity'] == 'I-TOOL')]
    # todo 临时的兼容性措施
    if argx_col_name is None:
        segs['igdt'] = direction_segment[
            (direction_segment['entity'] == 'B-EXPLICITINGREDIENT')
            | (direction_segment['entity'] == 'I-EXPLICITINGREDIENT')
            | (direction_segment['entity'] == 'B-IMPLICITINGREDIENT')
            | (direction_segment['entity'] == 'I-IMPLICITINGREDIENT')]
    else:
        argx_type = direction_segment[argx_col_name].apply(lambda x: x.split('-')[-1])
        argx_flag = (argx_type == 'Patient') | (argx_type == 'Theme')
        segs['igdt'] = direction_segment[
            ((direction_segment['entity'] == 'B-EXPLICITINGREDIENT')
             | (direction_segment['entity'] == 'I-EXPLICITINGREDIENT')
             | (direction_segment['entity'] == 'B-IMPLICITINGREDIENT')
             | (direction_segment['entity'] == 'I-IMPLICITINGREDIENT'))
            & argx_flag
            ]

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
def keywords_states_all_in_segment(keywords, sent_tokens):
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
        match_flag = keywords_states_all_in_segment(q_kws, direction_tokens + direction_tokens_lemma)
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
    ret_string = ret_string.replace('_', ' ')
    return ret_string


# 对于id重复的item，保留第一个，丢弃第二个
def filter_items_by_id(items: list):
    used_ids = []
    new_items = []
    for item in items:
        new_item, id = item.split('.', 1)
        if id in used_ids:
            continue
        else:
            used_ids.append(id)
            new_items.append(new_item)
    return new_items


# 提取segment内的所有item信息
def collect_segment_items(key_verb_row, segment, argx_col_name):
    entity_items = get_segment_entity_info(segment, argx_col_name=argx_col_name)
    hidden_items = parse_hidden(key_verb_row['hidden'], reserve_idx=True)
    coref_items = parse_coref(key_verb_row['coref'], reserve_idx=True)
    argx_items = {} if argx_col_name is None else get_segment_argx_info(segment, argx_col_name)
    ret_dict = {
        'entity': entity_items,
        'hidden': hidden_items,
        'coref': coref_items,
        'argx': argx_items,
    }
    return ret_dict


# 核心定位函数：根据【问题】，定位到相关上下文，并返回答案关键词及其它信息
def kernal_location_function(key_str_q, data_drt, data_drt_new):
    ret_dic_list = []
    # 根据核心文本，匹配和定位操作步骤
    matched_drts, q_kws = locate_direction(key_str_q, data_drt_new)
    # 遍历匹配到的所有操作步骤
    for seq_id, _ in matched_drts:
        # 提取核心动词 todo 第一个动词可能不是核心动词
        key_verb = key_str_q.split(' ')[0]
        direction = data_drt[data_drt['seq_id'] == seq_id]
        # 定位核心词的可能位置
        key_verb_idxs = get_keyword_loc(key_verb, direction)
        # 遍历核心动词的可能位置
        for key_verb_idx in key_verb_idxs:
            # 截取核心动词的关联上下文
            segment, col_name = locate_direction_segment(key_verb_idx, direction)
            # todo (done)检验keyword是否都在seg里，copy to another
            items = collect_annotation_items(segment)
            seg_tokens = segment['form'].tolist() + segment['lemma'].tolist()
            match_flag = keywords_states_all_in_segment(q_kws, [t for ts in items for t in ts] + seg_tokens)
            key_verb_row = direction.iloc[key_verb_idx]
            ret_dic = {'row': key_verb_row, 'seg': segment, 'drt': direction, 'q_kws': q_kws, 'argx_col': col_name}
            # 若问句与截取出的字段，相匹配
            if match_flag:
                key_verb_row = direction.iloc[key_verb_idx]
                item_infos = collect_segment_items(key_verb_row, segment, col_name)
                ret_dic.update(item_infos)
                ret_dic_list.append(ret_dic)
            # 若问句与截取出的字段，不相匹配
            else:
                continue
    return ret_dic_list


# 根据相应列的取值，截取操作步骤的片段
def get_conditional_segment(old_segment, col_name, col_values, qa_type, joint_rule='or'):
    flag = None
    for col_value in col_values:
        if joint_rule == 'or':
            flag = old_segment[col_name] == col_value if flag is None else flag | (old_segment[col_name] == col_value)
        elif joint_rule == 'and':
            flag = old_segment[col_name] == col_value if flag is None else flag & (old_segment[col_name] == col_value)
        else:
            raise Exception('Unknow joint_rule name!')
    if flag is None:
        if qa_type == 'act_ref_tool_or_full_act':
            new_segment = []
        else:
            print(qa_type)
            print(old_segment)
            raise Exception('get_conditional_segment return None!')
    else:
        new_segment = old_segment[flag]
    return new_segment


# 从原文中直接抽取答案的方法
def kernal_extract_answer_function(qa_type, key_str_q, data_drt, data_drt_new, question=None, answer=None):
    argx_tpye_map = {
        'act_ref_tool_or_full_act': ['Attribute', 'Instrument'],
        'act_igdt_ref_place': ['Destination', 'Location', 'Co-Patient'],
        'act_duration': ['Time'],
        'act_extent': ['Result'],
        'act_reason': ['Purpose', 'Cause'],
        'act_from_where': ['Source'],
        'act_couple_igdt': ['Co-Patient'],  # todo 是不是和act_igdt_ref_place一样？
        'igdt_amount': ['Extent'],
    }

    match_infos = kernal_location_function(key_str_q, data_drt, data_drt_new)
    pred_answers = []
    for info in match_infos:
        if info['argx_col'] is None:
            continue
        elif qa_type == 'act_ref_tool_or_full_act':
            # 抽取问题的谓语、宾语片段 todo 可能不含Co-Patient/Co-Theme类别
            q_extract_argx_types = ['B-V', 'I-V', 'D-V', 'B-Patient', 'I-Patient', 'B-Co-Patient', 'I-Co-Patient',
                                    'B-Theme', 'I-Theme', 'B-Co-Theme', 'I-Co-Theme']
            question_seg = get_conditional_segment(old_segment=info['seg'], col_name=info['argx_col'],
                                                   qa_type=qa_type, col_values=q_extract_argx_types)
            items = collect_annotation_items(question_seg)
            seg_tokens = question_seg['form'].tolist() + question_seg['lemma'].tolist()
            q_extract_tokens = []
            for token in key_str_q.split(' '):
                kws = [(token, lm.lemmatize(token, 'v'), lm.lemmatize(token, 'n'))]
                token_match_flag = keywords_states_all_in_segment(kws, [t for ts in items for t in ts] + seg_tokens)
                if token_match_flag:
                    q_extract_tokens.append(token)
            pred_answer_head = ' '.join(q_extract_tokens)
            # 抽取原文的Attribute片段
            attribute_argx_types = ['B-Attribute', 'I-Attribute']
            attribute_seg = get_conditional_segment(old_segment=info['seg'], col_name=info['argx_col'],
                                                    qa_type=qa_type, col_values=attribute_argx_types)
            attribute_str = ' '.join(attribute_seg['form'].tolist()) if len(attribute_seg) > 0 else None
            # 抽取原文的Instrument片段
            instrument_argx_types = ['B-Instrument', 'I-Instrument']
            instrument_seg = get_conditional_segment(old_segment=info['seg'], col_name=info['argx_col'],
                                                     qa_type=qa_type, col_values=instrument_argx_types)
            instrument_str = ' '.join(instrument_seg['form'].tolist()) if len(instrument_seg) > 0 else None
            # 判断是将Attribute如答案还是将Instrument拼接入答案
            if attribute_str is None and instrument_str is None:
                continue
            elif attribute_str is not None and instrument_str is None:
                pred_answer_tail = attribute_str
            elif attribute_str is None and instrument_str is not None:
                pred_answer_tail = instrument_str
            elif attribute_str is not None and instrument_str is not None:
                if key_str_q.__contains__(attribute_str) and key_str_q.__contains__(instrument_str):
                    raise Exception('key_str_q.__contains__(attribute_str) and key_str_q.__contains__(instrument_str)')
                if key_str_q.__contains__(attribute_str) and not key_str_q.__contains__(instrument_str):
                    pred_answer_tail = instrument_str
                elif not key_str_q.__contains__(attribute_str) and key_str_q.__contains__(instrument_str):
                    pred_answer_tail = attribute_str
                elif not key_str_q.__contains__(attribute_str) and not key_str_q.__contains__(instrument_str):
                    print('key_str_q not contain both attribute_str and instrument_str')
                    print('answer: {}\nattribute: {}\ninstrument: {}'.format(answer, attribute_str, instrument_str))
                    # todo 随便选一个attribute_str
                    pred_answer_tail = attribute_str
                else:
                    raise Exception('would not enter this if-else branch')
            else:
                raise Exception('if attribute_str is None and instrument_str is None')
            # 拼接答案
            pred_answer = pred_answer_head + ' ' + pred_answer_tail
            pred_answers.append(pred_answer)
        else:
            argx_base_types = argx_tpye_map[qa_type]
            argx_types = ['B-' + argx_base_type for argx_base_type in argx_base_types] \
                         + ['I-' + argx_base_type for argx_base_type in argx_base_types]
            seg = get_conditional_segment(old_segment=info['seg'], col_name=info['argx_col'], qa_type=qa_type,
                                          col_values=argx_types)
            if len(seg) > 0:
                pred_answer = ' '.join(seg['form'].tolist())
                pred_answers.append(pred_answer)
            else:
                continue
    return pred_answers


# 从标注知识中抽取答案的方法
def kernal_knowledge_answer_function(qa_type, key_str_q, data_drt, data_drt_new, question=None, answer=None):
    match_infos = kernal_location_function(key_str_q, data_drt, data_drt_new)
    pred_answers = []
    for info in match_infos:
        # if info['argx_col'] is None:
        if qa_type == 'act_ref_igdt':
            entity_igdt = info['entity']['igdt']
            hidden_drop = info['hidden'].get('drop', [])
            hidden_drop = filter_items_by_id(hidden_drop)
            # todo 没用到shadow？
            hidden_shadow = info['hidden'].get('shadow', [])
            hidden_shadow = filter_items_by_id(hidden_shadow)
            igdt = entity_igdt + hidden_drop
            if len(igdt) > 0:
                pred_answer = 'the ' + join_items(igdt)
                pred_answers.append(pred_answer)
            else:
                continue
        elif qa_type == 'act_ref_place':
            hidden_habitat = info['hidden'].get('habitat', [])
            hidden_habitat = filter_items_by_id(hidden_habitat)
            entity_habitat = info['entity']['habitat']
            if len(hidden_habitat) > 0:
                # todo 如果有多个habitat，先取第一个
                pred_answer = hidden_habitat[0].replace('_', ' ')
                pred_answers.append(pred_answer)
            elif len(entity_habitat) > 0:
                # todo 如果有多个habitat，先取第一个
                pred_answer = entity_habitat[0]
                pred_answers.append(pred_answer)
            else:
                continue
        elif qa_type == 'act_ref_tool_or_full_act':
            hidden_tool = info['hidden'].get('tool', [])
            hidden_tool = filter_items_by_id(hidden_tool)
            entity_tool = info['entity']['tool']
            if len(hidden_tool) > 0:
                # todo 如果有多个tool，先取第一个
                tool = hidden_tool[0].replace('_', ' ')
            elif len(entity_tool) > 0:
                # todo 如果有多个tool，先取第一个
                tool = entity_tool[0]
            else:
                continue
            if tool in ['hand', 'hands']:
                pred_answer = 'by hand'
            else:
                pred_answer = 'by using a {}'.format(tool).replace('_', ' ')
            pred_answers.append(pred_answer)
        else:
            raise Exception('unexpected qa_type in kernal_knowledge_answer_function: {}'.format(qa_type))

    return pred_answers


if __name__ == '__main__':
    print('END')
