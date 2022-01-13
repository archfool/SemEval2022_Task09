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
from init_config import src_dir, data_dir

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
    for tmp_dfs in [ingredient_dfs, direction_dfs]:
        for tmp_df in tmp_dfs:
            tmp_df['form'] = tmp_df['form'].apply(lambda x: x.strip())
            tmp_df['lemma'] = tmp_df['lemma'].apply(lambda x: x.strip())
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
    }
    return new_recipe


type_cat_id_dict = {
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
    'act_ref_igdt': ['and', ','],
    'act_ref_place': ['and', ','],
    'act_ref_tool': ['and', ','],
    'in_container': ['and', ','],
    'get_middle_igdt': ['and'],
    'act_extent': ['and'],
    'full_act': [],
    'act_duration': [],
    'add_igdt_place': [],
    'igdt_act_ref_place': [],
    'igdt_amount': [],
    'act_reason': [],
    'act_couple_igdt': ['and', ','],
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
    cat_id = qa_row['cat_id']

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
                    assert cat_id == 18
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
                        assert cat_id in type_cat_id_dict[qa_type]
                        key_str_q = regex_q.group(1)
                        key_str_a = regex_a.group(1)
                        keyword_a = [key_str_a]
                        for sep in type_sep_dict[qa_type]:
                            keyword_a = [y.strip() for x in keyword_a for y in x.split(sep)]
                        return question, answer, method, type, key_str_q, key_str_a, keyword_a
    # assert cat_id > 17
    print(qa_row, flush=True)
    raise ValueError('出现了意料之外的QA模式')


def inter_parse_qa_test(qa_df):
    for idx, qa_row in qa_df.iterrows():
        if qa_row['question'].__contains__("How do you shell the peas in the bowl?"):
            print(qa_row['question'])
        print('{}\t{}'.format(qa_row['question'], qa_row['answer']), flush=True)
        tmp = parse_qa(qa_row)


# 标注一条QA样本
def label_single_qa_sample(qa, recipe, sample):
    ingredients = recipe['ingredient_dfs']
    directions = recipe['direction_dfs']

    # 走规则模型
    if qa['type'] in ['count', 'act_first']:
        # todo 不标注
        return None

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


# 自动标注
def auto_label(recipe):
    # 判断对于同一个菜谱，是否存在同样问句
    assert len(recipe['qa_df']) == len(recipe['qa_df']['question'].drop_duplicates())

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
            'cat_id': qa['cat_id'],
            'type': qa['type'],
            'match_info': None,
        }
        sample = label_single_qa_sample(qa, recipe, sample)
        if sample is not None:
            samples.append(sample)
    return samples


# 菜谱信息分析
def recipe_analyze(recipes, mode):
    if False == mode:
        return

    ingredient_all = pd.concat([df for recipe in recipes.values() for df in recipe['ingredient_dfs']])
    for col in ['xpos', 'feats', 'head', 'deps', 'misc']:
        print("======{}======".format(col))
        print(ingredient_all[~pd.isna(ingredient_all[col])])
    for col in ['deprel']:
        print("======{}======".format(col))
        print(ingredient_all[ingredient_all[col] != '_'])

    direction_all = pd.concat([df for recipe in recipes.values() for df in recipe['direction_dfs']])
    for col in ['head']:
        print("======{}======".format(col))
        print(direction_all[~pd.isna(direction_all[col])])
    # xpos: EVENT, EXPLICITINGREDIENT, IMPLICITINGREDIENT, HABITAT, TOOL
    a = direction_all[~pd.isna(direction_all['feats'])]
    # deprel: Drop, Habitat, Tool, Result, Shadow
    dep = [dep for deps in direction_all[direction_all['deprel'] != '_']['deprel'].to_list() for dep in deps.split('|')]
    pd.Series([d.split('=')[0] for d in dep]).value_counts()
    c = direction_all[['form', 'deps']]
    d = direction_all[~pd.isna(direction_all['misc'])][['form', 'misc']]


def data_process(dataset_name):
    data_train_dir = os.path.join(data_dir, 'train')
    data_vali_dir = os.path.join(data_dir, 'val')
    data_test_dir = os.path.join(data_dir, 'test')
    # fields = ['id', 'from', 'lemma', 'upos', 'entity', 'part', 'hidden', 'coref', 'predicate', 'arg1']
    fields = None

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

    # 分析材料清单和操作步骤的额外信息
    recipe_analyze(recipes, True)

    # 自动标注
    all_samples = []
    for recipe_id, recipe in recipes.items():
        recipe_samples = auto_label(recipe)
        all_samples += recipe_samples
    for sample in all_samples:
        tks = sample['tokens']
        cur_offset = -1
        offset_maping = []
        for tk in tks:
            offset_maping.append((cur_offset + 1, cur_offset + 1 + len(tk)))
            cur_offset = cur_offset + 1 + len(tk)
        sample['context'] = ' '.join(tks)
        sample['offset_maping'] = offset_maping
    data_df = pd.DataFrame(all_samples)
    if False:
        data_df['len'] = data_df['tokens'].apply(len)
        for qa_type, tmp_df in data_df.groupby(['type']):
            print(qa_type)
            print(tmp_df['match_info'].value_counts(dropna=False))
            print("")
            a = data_df[data_df['type'] == 'full_act']
    dataset = {
        'question': data_df['question'].to_list(),
        'context': data_df['context'].to_list(),
        'label': data_df['label'].to_list(),
        'label_offset': data_df['offset_maping'].to_list(),
        'id': data_df['id'].to_list(),
        'answer': data_df['answer'].to_list(),
    }
    if False:
        qa_all = pd.concat([r['qa_df'] for r in recipes]).sort_values(['cat_id', 'seq_id']).reset_index(drop=True)
        # inter_parse_qa_test(qa_all)
        qa_all.to_csv(os.path.join(data_dir, 'qa_val.csv'), index=None, sep='\t', encoding='utf-8')
    return dataset


if __name__ == "__main__":
    dataset = data_process('vali')
    print('END')
