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
from datasets import load_metric
import json

lemmer = ns.WordNetLemmatizer()
# metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")
metric = load_metric("squad")

from init_config import src_dir, data_dir
from data_process import data_process, token_match, qa_type_rule, type_q_regex_pattern_dict, get_keywords, q_stopwords

with open(os.path.join(src_dir, 'present_tense.json'), 'r', encoding='utf-8') as f:
    present_tense_map = json.load(f)

"""
可以将操作的核心动词作为锚点。
entity列和hidden列是一组。在entity列被标记为B-EVENT。hidden列仅在entity=EVENT时，可能存在值。
upos列和argX列事一组。在upos列被标为VERB或其它。argX列仅在upos=VERB时，可能存在值。每列argX，有且仅有一个核心动词V。
entity=EVENT和upos=VERB存在一定的共现性。
"""


def load_dataset_file(dataset_name):
    print("\nLoading {} Dataset......".format(dataset_name.upper()))

    data_train_dir = os.path.join(data_dir, 'train')
    data_vali_dir = os.path.join(data_dir, 'val')
    data_test_dir = os.path.join(data_dir, 'test')

    if 'train' == dataset_name:
        data_uesd_dir = data_train_dir
    elif 'vali' == dataset_name:
        data_uesd_dir = data_vali_dir
    elif 'test' == dataset_name:
        data_uesd_dir = data_test_dir
    else:
        data_uesd_dir = None

    data_qa = pd.read_csv(os.path.join(data_uesd_dir, 'data_qa.csv'), sep='\x01', encoding='utf-8')
    data_drt = pd.read_csv(os.path.join(data_uesd_dir, 'data_direction.csv'), sep='\x01', encoding='utf-8')
    data_drtn = pd.read_csv(os.path.join(data_uesd_dir, 'data_direction_new.csv'), sep='\x01', encoding='utf-8')
    data_igdt = pd.read_csv(os.path.join(data_uesd_dir, 'data_ingredient.csv'), sep='\x01', encoding='utf-8')

    return data_qa, data_drt, data_drtn, data_igdt


def parse_hidden(hiddens, reserve_idx=False):
    if hiddens == '_':
        return {}

    hidden_dict = {hidden.split('=')[0]: [hid for hid in hidden.split('=')[1].split(':')]
                   for hidden in hiddens.split('|')}
    if reserve_idx is False:
        hidden_dict = {hid_name: [hid_value.split('.')[0] for hid_value in hid_values]
                       for hid_name, hid_values in hidden_dict.items()}
    return hidden_dict


def parse_direction_segment(direction_segment, anchor_verb_idx=None):
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


def parse_id(row):
    id = row['id']
    recipe_id, question_id, question = id.split('###')
    family_id = question_id.split('-')[0]
    return recipe_id, question_id


# 截取关键动词的关联上下文
def get_direction_segment(idx, direction):
    # 若关键动词，在argX列有标识，能够提取出上下文，则返回相关上下文
    for col_name in ['arg{}'.format(str(i)) for i in range(1, 11)]:
        if direction.iloc[idx][col_name] != '_' and 'V' == direction.iloc[idx][col_name].split('-')[1]:
            seg_df = direction[direction[col_name] != '_']
            return seg_df
    # 若关键动词，在argX列没有标识，不能够提取出上下文，则返回关键动词的当前行
    seg_df = direction[idx:idx + 1]
    return seg_df


# 连接item成完整的通顺句子
def join_items(item_list):
    # 若输入队列为空，则返回空字符串
    if len(item_list) == 0:
        return ''
    # 如果队列的元素还是队列，则将队列元素转换为字符串
    if isinstance(item_list[0], list):
        item_list = [' '.join(item) for item in item_list]
    # 按照口语习惯，拼接item
    ret_string = ' and '.join([x for x in [', '.join(item_list[:-1]), item_list[-1]] if x != ''])
    # ret_string = ret_string.replace('_', ' ')
    return ret_string


def collect_hidden(hiddens, target):
    lemme_target = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    collected_items = []
    for values in hiddens.tolist():
        value_list = [y for x in values.split('|') for y in x.split('=')[1].split(':')] if values != '_' else []
        for value in value_list:
            lemme_value = '_'.join([lemmer.lemmatize(token, 'n') for token in value.split('.')[0].split('_')])
            if lemme_value == lemme_target:
                collected_items.append(value)
    return collected_items


def collect_coref(corefs, target):
    lemme_target = '_'.join([lemmer.lemmatize(token, 'n') for token in target.split(' ') if token != ''])
    collected_items = []
    for values in corefs.tolist():
        for value in values.split(':'):
            lemme_value = '_'.join([lemmer.lemmatize(token, 'n') for token in value.split('.')[0].split('_')])
            if lemme_value == lemme_target:
                collected_items.append(value)
    return collected_items


def act_first(question, data_drt_new, verb_lemma_list):
    directions = data_drt_new
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
        if not (first[0][1] in verb_lemma_list and second[0][1] in verb_lemma_list):
            continue
        # 记录匹配到前半句和后半句的句子的idx
        first_seq_ids = []
        second_seq_ids = []
        # 遍历操作步骤的所有句子
        for seq_id, direction in directions.groupby('seq_id'):
            direction_tokens = direction['form'].tolist()
            direction_tokens_lemma = direction['lemma'].tolist()
            # 判断前半句问题，是否匹配到当前文本段落
            first_match_cnt = 0
            for f in first:
                if token_match(f, direction_tokens + direction_tokens_lemma):
                    first_match_cnt += 1
            if first_match_cnt == len(first):
                first_seq_ids.append(seq_id)
            # 判断后半句问题，是否匹配到当前文本段落
            second_match_cnt = 0
            for s in second:
                if token_match(s, direction_tokens + direction_tokens_lemma):
                    second_match_cnt += 1
            if second_match_cnt == len(second):
                second_seq_ids.append(seq_id)

        if len(first_seq_ids) > 0 and len(second_seq_ids) > 0:
            match_result[sep_idx] = (first_seq_ids, second_seq_ids)
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
    # 提取关键词(包括词元状态)
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
        # 遍历所有argX列，借助其信息，截取关联上下文
        for col_name in ['arg{}'.format(str(i)) for i in range(1, 11)]:
            # if row[col_name] in ['B-{}'.format(focus_type), 'I-{}'.format(focus_type)]:
            if row[col_name] != '_':
                # verb_rows = direction[direction[col_name] == 'B-V']
                # 截取关联的上下文
                target_rows = direction[
                    (direction[col_name] != '_') & (direction['entity'] == 'B-{}'.format(target_type))]

                # 未提取到信息
                if len(target_rows) == 0:
                    continue
                # 结合entity字段的HABITAT标注，从上下文token的coref中，提取place信息
                elif 'HABITAT' == target_type:
                    # places = set([place.split('.')[0]
                    #               for places in target_rows['coref'].tolist() if places != '_'
                    #               for place in places.split(':')])
                    places = set([place.split('.')[0] for place in target_rows['coref'].tolist() if place != '_'])
                    place = ' '.join([token for place in places for token in place.split('_')])
                    return place
                # 结合entity字段的EVENT标注，提取act信息
                elif 'EVENT' == target_type:
                    # todo 多行时，拼接结果，而不要找到一个就返回
                    for idx, target_row in target_rows.iterrows():
                        hiddens = parse_hidden(target_row['hidden'], False)
                        # 若hidden列包含Habitat字段信息，则直接取出place
                        if 'Habitat' in hiddens.keys():
                            # todo 多个Habitat时，拼接结果
                            place = hiddens['Habitat'][0].replace('_', ' ')
                            return place
                else:
                    raise ValueError("")

        return None

    # 4.往前反向搜索，找到igdt的place
    place = None
    # 获取igdt的词元字符串
    lemme_igdt = '_'.join([lemmer.lemmatize(token, 'n') for token in igdt.split(' ') if token != ''])
    # 遍历操作步骤的所有sents
    for igdt_sent_idx in range(sent_idx, -1, -1):
        old_direction = old_direction_dfs[igdt_sent_idx]
        start_token_idx = verb_token_idx if igdt_sent_idx == sent_idx else len(old_direction_dfs[igdt_sent_idx])
        # 遍历一条sent下的所有rows
        for igdt_token_idx in range(start_token_idx - 1, -1, -1):
            row = old_direction.iloc[igdt_token_idx]
            # 提取coref列的信息（词元格式）
            cur_coref_list = [item.split('.')[0] for item in row['coref'].split(':') if item != '_']
            lemma_cur_coref_list = ['_'.join([lemmer.lemmatize(token, 'n') for token in item.split('_')]) for item in
                                    cur_coref_list]
            # 提取hidden的信息（词元格式）
            hiddens = parse_hidden(row['hidden'], reserve_idx=False)
            cur_hidden_list = hiddens.get('Drop', []) + hiddens.get('Shadow', [])
            lemma_cur_hidden_list = ['_'.join([lemmer.lemmatize(token, 'n') for token in item.split('_')]) for item in
                                     cur_hidden_list]

            # 判断是否在coref列显式命中食材，同时当前词为名词igdt
            if lemme_igdt in lemma_cur_coref_list:
                # 尝试从原始上下文文本中，结合entity字段的HABITAT标注，获取显式位置信息
                habitat = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'Patient', 'HABITAT')
                # 原始文本中，显式包含位置信息
                if habitat is not None:
                    place = habitat
                # 原始文本中，不包含位置信息。则去上下文的动词处，寻找是否隐式包含place/HABITAT信息。
                else:
                    place = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'Patient', 'EVENT')
            # 判断是否在hidden列隐式命中食材，同时当前词为动词act
            elif lemme_igdt in lemma_cur_hidden_list:
                # 若hidden列包含Habitat字段信息，则直接取出place
                if 'Habitat' in hiddens.keys():
                    place = hiddens['Habitat'][0].replace('_', ' ')
                # 若hidden列未包含Habitat字段信息，则尝试从act的上下文文本中，结合entity字段的HABITAT标注，获取显式位置信息
                else:
                    place = get_igdt_or_act_from_arg_and_entity(row, old_direction, 'V', 'HABITAT')
            # 都没命中，进入进入下一个token的过程
            else:
                place = None

            # 如果命中place，则返回
            if place is not None:
                return place

    return place


def get_result_and_context(target_result, data_drt, tmp=None):
    target_result_tokens = [[token, lemmer.lemmatize(token, 'n'), token[:-1] if token.endswith('s') else token] for
                            token in target_result.split(' ')]
    # 遍历所有操作步骤
    for _, direction in data_drt.groupby('seq_id'):
        # 遍历所有token
        for r_idx in range(len(direction)):
            row = direction.iloc[r_idx]
            # 针对包含result信息的token，进行分析
            if row['hidden'] != '_' and 'Result' in parse_hidden(row['hidden']).keys():
                hiddens = parse_hidden(row['hidden'])
                results = hiddens['Result']
                # 遍历原文result
                for result in results:
                    match_flag = True
                    result_tokens = [(token, lemmer.lemmatize(token, 'n'), token[:-1] if token.endswith('s') else token)
                                     for token in result.split('_')]
                    # 原文result是否包含目标result
                    for target_result_token in target_result_tokens:
                        if not token_match(target_result_token, [t for ts in result_tokens for t in ts]):
                            match_flag = False
                            break
                    # 目标result是否包含原文result
                    for result_token in result_tokens:
                        if not token_match(result_token, [t for ts in target_result_tokens for t in ts]):
                            match_flag = False
                            break
                    # 目标result匹配到原文result
                    if match_flag:
                        seg_df = get_direction_segment(r_idx, direction)
                        seg_infos = parse_direction_segment(seg_df, direction.index[r_idx])
                        return hiddens, seg_infos
            else:
                pass
    return None, None


# def f1_metric(row):
#     predictions = [{'id': 0, 'prediction_text': row['pred_answer']}]
#     references = [{'id': 0, 'answers': {'text': [row['answer']], 'answer_start': [0]}}]
#     metric_result = metric.compute(predictions=predictions, references=references)
#     return metric_result['f1']


# def tmp_metric(row):
#     if row['qa_type'] in ['act_first', 'place_before_act', 'count_times', 'count_nums']:
#         return 1 if str(row['answer']) == str(row['pred_answer']) else 0
#     else:
#         return 0


# 加载处理好的数据文件


def rule_for_qa(data_qa, data_drt, data_drt_new, data_igdt):
    verb_lemma_list = data_drt_new[data_drt_new['upos'] == 'VERB']['lemma'].tolist()
    def get_answer_by_rule(row, data_drt=data_drt, data_drt_new=data_drt_new, data_igdt=data_igdt):
        recipe_id = row['recipe_id']
        qa_type = row['qa_type']
        question = row['question']
        answer = row['answer']
        key_context = row['context']  # 问题的关键字段
        data_drt = data_drt[data_drt['recipe_id'] == recipe_id]
        data_drt_new = data_drt_new[data_drt_new['recipe_id'] == recipe_id]
        data_igdt = data_igdt[data_igdt['recipe_id'] == recipe_id]
        # direction_dfs = recipe['direction_dfs']
        # new_direction_dfs = recipe['new_direction_dfs']
        # directions = pd.concat(direction_dfs)

        if 'act_first' == qa_type:
            ret = act_first(key_context, data_drt_new, verb_lemma_list)
            return ret
        # elif 'place_before_act' == qa_type:
        #     igdt, act = key_context.split('|')
        #     place = place_before_act(igdt, act, new_direction_dfs, direction_dfs)
        #     return 'N/A' if place is None else place
        # elif 'count_times' == qa_type:
        #     collected_hidden = collect_hidden(directions['hidden'], key_context)
        #     collected_coref = collect_coref(directions['coref'], key_context)
        #     count = len(collected_hidden + collected_coref)
        #     count = 'N/A' if count == 0 else str(count)
        #     return count
        # elif 'count_nums' == qa_type:
        #     collected_hidden = collect_hidden(directions['hidden'], key_context)
        #     collected_coref = collect_coref(directions['coref'], key_context)
        #     num = len(set(collected_hidden + collected_coref))
        #     num = 'N/A' if num == 0 else str(num)
        #     return num
        elif 'get_result' == qa_type:
            hiddens, seg_infos = get_result_and_context(key_context, data_drt, answer)
            if hiddens is not None:
                # 动作字符串
                act_string = ' '.join([present_tense_map[act] for act in seg_infos['act']])
                # act_string = conjugate(verb=act, tense=PARTICIPLE, number=SG)
                act_string = 'by ' + act_string if act_string != '' else None
                # 食材字符串
                igdts = seg_infos['igdt'] + hiddens.get('Drop', []) + hiddens.get('Shadow', [])
                igdt_string = join_items(igdts)
                igdt_string = 'the ' + igdt_string if igdt_string != '' else None
                # 容器字符串
                hibatit = seg_infos['habitat'] + [hiddens['Habitat'][0]] if hiddens.__contains__('Habitat') else []
                hibatit_string = join_items(hibatit)
                hibatit_string = 'in the ' + hibatit_string if hibatit_string != '' else None
                # 工具字符串
                tool = seg_infos['tool'] + [hiddens['Tool'][0]] if hiddens.__contains__('Tool') else []
                tool_string = join_items(tool)
                tool_string = 'with the ' + tool_string if tool_string != '' else None
                # 汇总各个字符串
                ret_string = ' '.join(
                    [s for s in [act_string, igdt_string, hibatit_string, tool_string] if s is not None])
                return ret_string.replace('_', ' ')
            else:
                return 'N/A'
        # elif 'result_component' == qa_type:
        #     hiddens, seg_infos = serch_result_infos(key_context, direction_dfs, answer)
        #     if hiddens is not None:
        #         igdts = seg_infos['igdt'] + hiddens.get('Drop', []) + hiddens.get('Shadow', [])
        #         igdt_string = join_items(igdts)
        #         igdt_string = 'the ' + igdt_string if igdt_string != '' else None
        #         ret_string = igdt_string if igdt_string is not None else 'N/A'
        #         return ret_string.replace('_', ' ')
        #     else:
        #         return 'N/A'
        else:
            return None
            # raise ValueError('invalid rule qa_type')

    data_qa[['recipe_id', 'question_id']] = data_qa.apply(parse_id, axis=1, result_type="expand")
    if os.path.exists(u'/media/archfool/data'):
        data_qa['pred_answer'] = data_qa.apply(get_answer_by_rule, axis=1)
    else:
        data_qa['pred_answer'] = None
        for idx in range(len(data_qa)):
            data_qa['pred_answer'].iloc[idx] = get_answer_by_rule(data_qa.iloc[idx])

    return data_qa


if __name__ == '__main__':
    # 加载数据
    if True:
        data_qa, data_drt, data_drt_new, data_igdt = load_dataset_file('vali')
    else:
        dataset_model_vali, dataset_rule_vali = data_process('vali')

    # 通过规则获取答案
    rule_result = rule_for_qa(data_qa, data_drt, data_drt_new, data_igdt)

    # 计算分数
    rule_result['score'] = rule_result.apply(lambda r: 1 if r['answer'] == r['pred_answer'] else 0, axis=1)
    # rule_result['score'] = rule_result.apply(f1_metric, axis=1)
    # rule_result['tmp_score'] = rule_result.apply(tmp_metric, axis=1)

    print(rule_result['qa_type'].value_counts(normalize=True))
    print('========== score: {} =========='.format(round(rule_result['score'].mean(), 2)))
    for qa_type in data_qa['qa_type'].value_counts().index.to_list():
        print('=========={}=========='.format(qa_type))
        print(rule_result[rule_result['qa_type'] == qa_type]['score'].value_counts(normalize=True).sort_index())
    # print('==========rule_module f1: {}=========='.format(round(rule_result['score'].mean(), 2)))
    # for qa_type in ['act_first', 'place_before_act', 'count_times', 'count_nums']:
    #     print('=========={}=========='.format(qa_type))
    #     print(rule_result[rule_result['qa_type'] == qa_type]['tmp_score'].value_counts(normalize=True).sort_index())
    # for qa_type in ['get_result', 'result_component']:
    #     print('=========={}=========='.format(qa_type))
    #     print(rule_result[rule_result['qa_type'] == qa_type]['score'].mean())

    print('END')
