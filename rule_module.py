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

lm = ns.WordNetLemmatizer()
# metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")
metric = load_metric("squad")

from init_config import src_dir, data_dir
from data_process import data_process, token_match, qa_type_rule, type_q_regex_pattern_dict, get_keywords, q_stopwords
from rule_utils import parse_id, parse_hidden, get_segment_entity_info, get_segment_argx_info, \
    keywords_states_all_in_segment, locate_direction, locate_direction_segment, get_keyword_loc, join_items, \
    collect_hidden, collect_coref, get_conditional_segment, collect_annotation_items, collect_segment_items, \
    kernal_location_function, filter_items_by_id, kernal_extract_answer_function, kernal_knowledge_answer_function, \
    locate_direction_segment_plus

with open(os.path.join(src_dir, 'present_tense.json'), 'r', encoding='utf-8') as f:
    present_tense_map = json.load(f)

"""
可以将操作的核心动词作为锚点。
entity列和hidden列是一组。在entity列被标记为B-EVENT。hidden列仅在entity=EVENT时，可能存在值。
upos列和argX列事一组。在upos列被标为VERB或其它。argX列仅在upos=VERB时，可能存在值。每列argX，有且仅有一个核心动词V。
entity=EVENT和upos=VERB存在一定的共现性。
"""


# def f1_metric(row):
#     predictions = [{'id': 0, 'prediction_text': row['pred_answer']}]
#     references = [{'id': 0, 'answers': {'text': [row['answer']], 'answer_start': [0]}}]
#     metric_result = metric.compute(predictions=predictions, references=references)
#     return metric_result['f1']
#
#
# def tmp_metric(row):
#     if row['qa_type'] in ['act_first', 'place_before_act', 'count_times', 'count_nums']:
#         return 1 if str(row['answer']) == str(row['pred_answer']) else 0
#     else:
#         return 0


def act_ref_tool_or_full_act_bak220131(qa_type, key_str_q, data_drt, data_drt_new, question=None, answer=None):
    # 尝试以qa_type为full_act获取答案
    pred_answers = kernal_extract_answer_function(qa_type, key_str_q, data_drt, data_drt_new, question, answer)
    if len(pred_answers) > 0:
        return pred_answers[0]
    # 根据核心文本，匹配和定位操作步骤
    matched_drts, q_kws = locate_direction(key_str_q, data_drt_new)
    # tools = []
    # attributes = []
    # 遍历匹配到的所有操作步骤
    for seq_id, _ in matched_drts:
        # 提取核心动词
        key_verb = key_str_q.split(' ')[0]
        direction = data_drt[data_drt['seq_id'] == seq_id]
        # 定位核心词的可能位置
        key_verb_idxs = get_keyword_loc(key_verb, direction)
        for key_verb_idx in key_verb_idxs:
            # todo 检验keyword是否都在seg里
            # 截取关键字段
            seg_df, col_name = locate_direction_segment(key_verb_idx, direction)
            # 若关键动词没有关联的上下文，则跳过
            if col_name is None:
                continue
            else:
                # 提取信息
                hiddens = parse_hidden(direction.iloc[key_verb_idx]['hidden'])
                entity_infos = get_segment_entity_info(seg_df)
                argx_infos = get_segment_argx_info(seg_df, col_name)
                # 整理信息
                vs = argx_infos['v']
                # todo patient和drop，shadow的顺序
                # todo 原文字段，使用argx_infos['patient']还是seg_infos['igdt']？
                igdts = argx_infos['patient'] + hiddens.get('drop', []) + hiddens.get('shadow', [])
                tools = entity_infos['tool'] + hiddens.get('tool', [])
                # todo，attribute/instrument有先后顺序，先不管了
                extras = argx_infos['attribute'] + argx_infos['instrument']
                # 判断属于哪个提问模板
                if len(tools) == 0 and len(extras) == 0:
                    continue
                elif len(tools) > 0 and len(extras) == 0:
                    qa_type = 'act_ref_tool'
                elif len(tools) == 0 and len(extras) > 0:
                    qa_type = 'full_act'
                elif len(tools) > 0 and len(extras) > 0:
                    keywords = get_keywords([key_str_q], seps=[' and ', ' '], stopwords=q_stopwords,
                                            puncts=['.', ',', ';'])
                    keywords = [(kw, lm.lemmatize(kw, 'v'), lm.lemmatize(kw, 'n')) for kw in keywords]
                    v_and_n_tokens = [token for tokens in vs + igdts for token in tokens.replace('_', ' ').split(' ')]
                    if keywords_states_all_in_segment(keywords, v_and_n_tokens):
                        qa_type = 'act_ref_tool'
                    else:
                        qa_type = 'full_act'
                else:
                    raise ValueError('whould not in this branch')

                # 使用by using a tool形式展示答案
                if qa_type == 'act_ref_tool':
                    # todo 如果有多个tool，先取第一个
                    tool = tools[0]
                    if tool in ['hand', 'hands']:
                        pred_answer = 'by hand'
                    else:
                        pred_answer = 'by using a {}'.format(tool).replace('_', ' ')
                    return pred_answer
                # 使用 v + igdt + extras 拼接答案
                elif qa_type == 'full_act':
                    # todo 如果有多个v，先取第一个
                    v = vs[0] if len(vs) > 0 else None
                    # 将v的第一个词替换为正常时态
                    v = ' '.join([lm.lemmatize(x, 'v') if i == 0 else x for i, x in enumerate(v.split(' '))])
                    igdt_s = join_items(igdts)
                    igdt_s = 'the ' + igdt_s if igdt_s != '' else None
                    extras_s = ' '.join(extras)
                    pred_answer = ' '.join([x for x in [v, igdt_s, extras_s] if x is not None]).replace('_', ' ')
                    return pred_answer
                else:
                    raise Exception('qa_type of act_ref_tool_or_full_act is unknow!')

    # 什么都没匹配到，返回N/A
    return 'N/A'


def place_before_act_bak220131(igdt, act, new_direction_dfs, old_direction_dfs):
    # 1.重定位igdt名字

    # 2.找到同时包含igdt和act的句子
    # 提取关键词(包括词元状态)
    keywords = get_keywords([igdt, act], seps=[',', ' '], stopwords=q_stopwords, puncts=['.', ';'])
    keywords = [(kw, lm.lemmatize(kw, 'v'), lm.lemmatize(kw, 'n')) for kw in keywords]
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
        verb_keyword = lm.lemmatize(verb_keyword, 'v')
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
    lemme_igdt = '_'.join([lm.lemmatize(token, 'n') for token in igdt.split(' ') if token != ''])
    # 遍历操作步骤的所有sents
    for igdt_sent_idx in range(sent_idx, -1, -1):
        old_direction = old_direction_dfs[igdt_sent_idx]
        start_token_idx = verb_token_idx if igdt_sent_idx == sent_idx else len(old_direction_dfs[igdt_sent_idx])
        # 遍历一条sent下的所有rows
        for igdt_token_idx in range(start_token_idx - 1, -1, -1):
            row = old_direction.iloc[igdt_token_idx]
            # 提取coref列的信息（词元格式）
            cur_coref_list = [item.split('.')[0] for item in row['coref'].split(':') if item != '_']
            lemma_cur_coref_list = ['_'.join([lm.lemmatize(token, 'n') for token in item.split('_')]) for item in
                                    cur_coref_list]
            # 提取hidden的信息（词元格式）
            hiddens = parse_hidden(row['hidden'], reserve_idx=False)
            cur_hidden_list = hiddens.get('drop', []) + hiddens.get('shadow', [])
            lemma_cur_hidden_list = ['_'.join([lm.lemmatize(token, 'n') for token in item.split('_')]) for item in
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


def act_first(question, data_drt_new):
    question = question.lower().replace('(', '\(').replace(')', '\)')
    # tokens_list = [[token for token in tokens.split(' ') if token != ''] for tokens in question.split(' and ')]
    tokens_list = [tokens for tokens in question.split(' and ')]
    tokens_list = [get_keywords([ts], seps=[',', ' '], stopwords=q_stopwords, puncts=['.', ';']) for ts in tokens_list]
    tokens_list = [[(tk, lm.lemmatize(tk, 'v'), lm.lemmatize(tk, 'n')) for tk in tks] for tks in tokens_list]

    match_result = {}
    # 遍历问题的前后半句的所有组合方式
    for sep_idx in range(1, len(tokens_list)):
        # 分割问题为前半句和后半句
        first = [token for tokens in tokens_list[:sep_idx] for token in tokens]
        second = [token for tokens in tokens_list[sep_idx:] for token in tokens]
        # 判断前半句和后半句的首词，是否为动词
        verb_list = data_drt_new[data_drt_new['upos'] == 'VERB']['lemma'].tolist()
        if not (first[0][1] in verb_list and second[0][1] in verb_list):
            continue
        # 记录匹配到前半句和后半句的句子的idx
        first_sent_idxs = []
        second_sent_idxs = []
        # 遍历操作步骤的所有句子
        # for sent_idx, direction in enumerate(direction_dfs):
        for seq_id, direction in data_drt_new.groupby('seq_id'):
            direction_tokens = direction['form'].tolist()
            direction_tokens_lemma = direction['lemma'].tolist()
            # 判断前半句问题，是否匹配到当前文本段落
            first_match_cnt = 0
            for f in first:
                if token_match(f, direction_tokens + direction_tokens_lemma):
                    first_match_cnt += 1
            if first_match_cnt == len(first):
                first_sent_idxs.append(seq_id)
            # 判断后半句问题，是否匹配到当前文本段落
            second_match_cnt = 0
            for s in second:
                if token_match(s, direction_tokens + direction_tokens_lemma):
                    second_match_cnt += 1
            if second_match_cnt == len(second):
                second_sent_idxs.append(seq_id)

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


def serch_result_infos(target_result, data_drt, question=None, answer=None):
    target_result_tokens = [[token, lm.lemmatize(token, 'n'), token[:-1] if token.endswith('s') else token] for
                            token in target_result.split(' ')]
    ret_dic_list = []
    # 遍历所有操作步骤
    for seq_id, direction in data_drt.groupby('seq_id'):
        # 遍历所有token
        for r_idx, row in direction.iterrows():
            results = parse_hidden(row['hidden']).get('result', [])
            # 遍历检索到的result
            for result in results:
                match_flag = True
                result_tokens = [(token, lm.lemmatize(token, 'n'), token[:-1] if token.endswith('s') else token)
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
                    segment, col_name = locate_direction_segment(r_idx, direction)
                    item_infos = collect_segment_items(row, segment, col_name, direction.index[r_idx])
                    ret_dic = {'row': row, 'seg': segment, 'drt': direction, 'argx_col': col_name}
                    ret_dic.update(item_infos)
                    ret_dic_list.append(ret_dic)
                else:
                    continue
    return ret_dic_list


def rule_for_qa(dataset):
    qa_df = dataset['qa_data']
    recipes = dataset['recipe_data']

    def get_answer_by_rule(row, recipes=recipes):
        recipe = recipes[row['recipe_id']]
        qa_type = row['qa_type']
        question = row['question']
        answer = row['answer']
        key_str_q = row['key_str_q']
        direction_dfs = recipe['direction_dfs']
        new_direction_dfs = recipe['new_direction_dfs']
        data_drt = recipe['data_drt']
        data_drt_new = recipe['data_drt_new']
        igdts = recipe['data_igdt']
        directions = pd.concat(direction_dfs)

        if qa_type in ['act_ref_igdt', 'act_ref_place']:
            pred_answers = kernal_knowledge_answer_function(qa_type, key_str_q, data_drt, data_drt_new, question,
                                                            answer)
            if len(pred_answers) != 0:
                # todo 暂时只取第一个答案
                return pred_answers[0]
            else:
                return 'N/A'
        elif qa_type in ['act_igdt_ref_place', 'act_duration', 'act_extent', 'act_reason', 'act_from_where',
                         'act_couple_igdt', 'igdt_amount']:
            pred_answers = kernal_extract_answer_function(qa_type, key_str_q, data_drt, data_drt_new, question, answer)
            if len(pred_answers) != 0:
                # todo 暂时只取第一个答案
                return pred_answers[0]
            else:
                return 'N/A'
        elif 'act_ref_tool_or_full_act' == qa_type:
            # 尝试在qa_type==full_act的前提下，获取答案
            pred_answers_extract = kernal_extract_answer_function(qa_type, key_str_q, data_drt, data_drt_new,
                                                                  question, answer)
            # 尝试在qa_type==act_ref_tool的前提下，获取答案
            pred_answers_knowledge = kernal_knowledge_answer_function(qa_type, key_str_q, data_drt, data_drt_new,
                                                                      question, answer)
            if len(pred_answers_extract) > 0:
                pred_answers = pred_answers_extract
            elif len(pred_answers_knowledge) > 0:
                pred_answers = pred_answers_knowledge
            else:
                pred_answers = []
            if len(pred_answers) != 0:
                # todo 暂时只取第一个答案
                return pred_answers[0]
            else:
                return 'N/A'
        elif 'act_first' == qa_type:
            ret = act_first(key_str_q, data_drt_new)
            return ret
        elif 'place_before_act' == qa_type:
            igdt, act = key_str_q.split('|')
            q_kws = get_keywords([key_str_q.replace('|', ' ')], seps=[' and ', ' '], stopwords=q_stopwords,
                                 puncts=['.', ',', ';'])
            q_kws = [(kw, lm.lemmatize(kw, 'v'), lm.lemmatize(kw, 'n')) for kw in q_kws]
            q_match_flag = False
            for idx, row in data_drt[::-1].iterrows():
                segment, argx_col = locate_direction_segment_plus(idx, row, data_drt[::-1])
                if segment is not None:
                    if q_match_flag is False:
                        items = collect_annotation_items(segment)
                        seg_tokens = segment['form'].tolist() + segment['lemma'].tolist()
                        q_match_flag = keywords_states_all_in_segment(q_kws,
                                                                      [t for ts in items for t in ts] + seg_tokens)
                    else:
                        item_infos = collect_segment_items(row, segment, argx_col)
                        entity_igdt = item_infos['entity']['igdt']
                        entity_habitat = item_infos['entity']['habitat']
                        hidden_drop = [x.split('.')[0].replace('_', ' ') for x in item_infos['hidden'].get('drop', [])]
                        hidden_shadow = [x.split('.')[0].replace('_', ' ') for x in
                                         item_infos['hidden'].get('shadow', [])]
                        hidden_habitat = [x.split('.')[0].replace('_', ' ') for x in
                                          item_infos['hidden'].get('habitat', [])]
                        coref = [x.split('.')[0].replace('_', ' ') for x in item_infos['coref']]
                        igdts = entity_igdt + hidden_drop + hidden_shadow + coref
                        habitats = entity_habitat + hidden_habitat
                        for i in igdts:
                            if igdt in [i, lm.lemmatize(i, 'n')] \
                                    or lm.lemmatize(igdt, 'n') in [i, lm.lemmatize(i, 'n')]:
                                if len(habitats) > 0:
                                    return habitats[0]
                else:
                    continue
            return 'N/A'
        elif 'count_times' == qa_type:
            collected_hidden = collect_hidden(directions['hidden'], key_str_q)
            collected_coref = collect_coref(directions['coref'], key_str_q)
            count = len(collected_hidden + collected_coref)
            count = 'N/A' if count == 0 else str(count)
            return count
        elif 'count_nums' == qa_type:
            collected_hidden = collect_hidden(directions['hidden'], key_str_q)
            collected_coref = collect_coref(directions['coref'], key_str_q)
            num = len(set(collected_hidden + collected_coref))
            num = 'N/A' if num == 0 else str(num)
            return num
        elif 'get_result' == qa_type:
            serch_infos = serch_result_infos(key_str_q, data_drt, question, answer)
            pred_answers = []
            for info in serch_infos:
                # 动作字符串
                entity_act = [present_tense_map[act] for act in info['entity']['act']]
                act_string = 'by ' + ' '.join(entity_act) if len(entity_act) > 0 else None
                # 食材字符串
                # todo part1不是数字的，要剔除
                entity_igdt = info['entity']['igdt']
                hidden_drop = info['hidden'].get('drop', [])
                hidden_drop = filter_items_by_id(hidden_drop)
                igdts = entity_igdt + hidden_drop
                igdt_string = 'the ' + join_items(igdts) if len(igdts) > 0 else None
                # 容器字符串:
                entity_habitat = info['entity']['habitat']  # 等于1的
                hidden_habitat = info['hidden'].get('habitat', [])
                hidden_habitat = filter_items_by_id(hidden_habitat)
                habitats = entity_habitat + hidden_habitat
                # todo 介词不全是in
                habitat_string = 'in the ' + join_items(habitats) if len(habitats) > 0 else None
                # 工具字符串
                entity_tool = info['entity']['tool']
                hidden_tool = info['hidden'].get('tool', [])
                hidden_tool = filter_items_by_id(hidden_tool)
                tools = entity_tool + hidden_tool
                tool_string = 'with the ' + join_items(tools) if len(tools) > 0 else None
                # 汇总各个字符串
                sub_strings = [s for s in [act_string, igdt_string, habitat_string, tool_string] if s is not None]
                if len(sub_strings) > 0:
                    pred_answer = ' '.join(sub_strings)
                    pred_answers.append(pred_answer)
                else:
                    continue
            if len(pred_answers) != 0:
                # todo 暂时只取第一个答案
                return pred_answers[0]
            else:
                return 'N/A'
            # for info in serch_infos:
            #     hiddens = info['hidden']
            #     seg_infos = info['entity']
            #     if hiddens is not None:
            #         # 动作字符串
            #         act_string = ' '.join([present_tense_map[act] for act in seg_infos['act']])
            #         # act_string = conjugate(verb=act, tense=PARTICIPLE, number=SG)
            #         act_string = 'by ' + act_string if act_string != '' else None
            #         # 食材字符串
            #         # todo 原文字段，使用argx_infos['patient']还是seg_infos['igdt']？
            #         igdts = seg_infos['igdt'] + hiddens.get('drop', []) + hiddens.get('shadow', [])
            #         igdt_string = join_items(igdts)
            #         igdt_string = 'the ' + igdt_string if igdt_string != '' else None
            #         # 容器字符串
            #         hibatit = seg_infos['habitat'] + [hiddens['habitat'][0]] if hiddens.__contains__('habitat') else []
            #         hibatit_string = join_items(hibatit)
            #         hibatit_string = 'in the ' + hibatit_string if hibatit_string != '' else None
            #         # 工具字符串
            #         tools = seg_infos['tool'] + [hiddens['tool'][0]] if hiddens.__contains__('tool') else []
            #         tool_string = join_items(tools)
            #         tool_string = 'with the ' + tool_string if tool_string != '' else None
            #         # 汇总各个字符串
            #         ret_string = ' '.join(
            #             [s for s in [act_string, igdt_string, hibatit_string, tool_string] if s is not None])
            #         return ret_string.replace('_', ' ')
            #     else:
            #         return 'N/A'
        elif 'result_component' == qa_type:
            serch_infos = serch_result_infos(key_str_q, data_drt, question, answer)
            pred_answers = []
            for info in serch_infos:
                entity_igdt = info['entity']['igdt']
                hidden_drop = info['hidden'].get('drop', [])
                hidden_drop = filter_items_by_id(hidden_drop)
                igdts = entity_igdt + hidden_drop
                if len(igdts) > 0:
                    pred_answer = 'the ' + join_items(igdts)
                    pred_answers.append(pred_answer)
                else:
                    continue
            if len(pred_answers) != 0:
                # todo 暂时只取第一个答案
                return pred_answers[0]
            else:
                return 'N/A'
            # hiddens = info['hidden']
            # seg_infos = info['entity']
            # if hiddens is not None:
            #     # todo 原文字段，使用argx_infos['patient']还是seg_infos['igdt']？
            #     igdts = seg_infos['igdt'] + hiddens.get('drop', []) + hiddens.get('shadow', [])
            #     igdt_string = join_items(igdts)
            #     igdt_string = 'the ' + igdt_string if igdt_string != '' else None
            #     ret_string = igdt_string if igdt_string is not None else 'N/A'
            #     return ret_string.replace('_', ' ')
            # else:
            #     return 'N/A'
        else:
            raise Exception('invalid rule qa_type: {}'.format(qa_type))

    qa_df[['recipe_id', 'question_id']] = qa_df.apply(parse_id, axis=1, result_type="expand")

    qa_df['pred_answer'] = qa_df.apply(get_answer_by_rule, axis=1)
    # qa_df['pred_answer'] = None
    # for idx in range(len(qa_df)):
    #     qa_df.iloc[idx]['pred_answer'] = get_answer_by_rule(qa_df.iloc[idx])

    return qa_df


if __name__ == '__main__':
    # 加载数据
    dataset_name = 'vali'
    dataset_model_vali, dataset_rule_vali = data_process(dataset_name)

    # 通过规则获取答案
    rule_result = rule_for_qa(dataset_rule_vali)

    # 计算分数
    rule_result['score'] = rule_result.apply(lambda r: 1 if r['answer'] == r['pred_answer'] else 0, axis=1)
    # rule_result['score'] = rule_result.apply(f1_metric, axis=1)
    # rule_result['tmp_score'] = rule_result.apply(tmp_metric, axis=1)

    # print(rule_result['qa_type'].value_counts(normalize=True))
    print('========== score: {}=========='.format(round(rule_result['score'].mean(), 4)))
    for qa_type in rule_result['qa_type'].value_counts().index.to_list():
        print('=========={}=========='.format(qa_type))
        print('qa_type per: {}%'.format(
            round(len(rule_result[rule_result['qa_type'] == qa_type]) / len(rule_result) * 100, 2)))
        print(rule_result[rule_result['qa_type'] == qa_type]['score'].value_counts(normalize=True).sort_index())
    # for qa_type in ['act_first', 'place_before_act', 'count_times', 'count_nums']:
    #     print('=========={}=========='.format(qa_type))
    #     print(rule_result[rule_result['qa_type'] == qa_type]['tmp_score'].value_counts(normalize=True).sort_index())
    # for qa_type in ['get_result', 'result_component']:
    #     print('=========={}=========='.format(qa_type))
    #     print(rule_result[rule_result['qa_type'] == qa_type]['score'].mean())

    if dataset_name == 'test':
        rule_result['pred_answer'] = rule_result['pred_answer'].apply(lambda x: None if (x == 'N/A' or x == '') else x)
        from task9_main import convert_pred_result_to_submission_format

        r2vq_pred_result = convert_pred_result_to_submission_format(rule_result)
        submit_filename = 'r2vq_pred.json'
        with open(os.path.join(src_dir, submit_filename), 'w', encoding='utf-8') as f:
            json.dump(r2vq_pred_result, f)

    print('END')
