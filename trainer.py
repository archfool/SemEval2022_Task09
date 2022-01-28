#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import sys
from dataclasses import dataclass, field
import collections
import json
import logging
import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from BertForExtractQA import BertForExtractQA
from data_process import upos_map, entity_map
import utils

# sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'framework'))
from util_tools import get_now_date_str, get_now_time_str
from util_data import tag_offset_mapping

import datasets
from datasets import load_dataset, load_metric

import torch
from torch import nn

cross_entropy_fct = nn.CrossEntropyLoss()

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version
from transformers.utils.versions import require_version

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds, eval_loss = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # 打印评估结果至文件
            filename = 'metric_record.log'
            with open(os.path.join(self.args.output_dir, filename), 'a+', encoding='utf-8') as f:
                record = {
                    'epoch': round(self.state.epoch, 2) if self.is_in_train else -1,
                    'step': self.state.global_step,
                    'loss': round(eval_loss, 6) if eval_loss is not None else None,
                    'f1': round(metrics['f1'], 2),
                    'exact_match': round(metrics['exact_match'], 2),
                }
                json.dump(record, f)
                f.write('\n')
                print(record, flush=True)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
            print(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions, loss = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")

        id_qa_type_map = {idx: qa_type for idx, qa_type in zip(predict_examples['id'], predict_examples['qa_type'])}
        for idx, prediction in enumerate(predictions):
            recipe_id, question_id, question = prediction['id'].split('###')
            qa_type = id_qa_type_map[prediction['id']]
            prediction['recipe_id'] = recipe_id
            prediction['question_id'] = question_id
            prediction['question'] = question
            prediction['qa_type'] = qa_type
            predictions[idx] = prediction
        ret_df = pd.DataFrame(predictions)

        return ret_df

        # metrics = self.compute_metrics(predictions)
        #
        # # Prefix all keys with metric_key_prefix + '_'
        # for key in list(metrics.keys()):
        #     if not key.startswith(f"{metric_key_prefix}_"):
        #         metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        # return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)


def postprocess_qa_predictions(
        examples,
        features,
        predictions,
        version_2_with_negative: bool = True,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
        tokenizer=None
):
    # 映射菜谱ID和真实答案/token偏置/标签
    id_answer_dict = {}
    id_offset_maping_dict = {}
    id_label_dict = {}
    for example in examples:
        example_recipe_id = example['id']
        answer = example['answer']
        offset_maping = example['offset_maping']
        label = example['label']
        id_answer_dict[example_recipe_id] = answer
        id_offset_maping_dict[example_recipe_id] = offset_maping
        id_label_dict[example_recipe_id] = label

    # 建立以菜谱ID为key的字典，value预测的答案关键字id集合
    id_pred_token_dict = {}
    # 所有样本的loss总和
    loss_list = []
    # 遍历所有训练样本
    for feature, prediction in zip(features, predictions):
        feature_recipe_id = feature['recipe_id']
        token_ids = feature['input_ids']
        token_type_ids = feature['token_type_ids']
        token_offsets = feature['token_offset']
        # todo prediction的格式可能会变动
        logits = prediction

        # 答案关键字集合
        pred_answer_token_list = []
        # 计算预测的label
        pred_labels = logits.argmax(axis=1).tolist()
        # tmp: pred_labels = [1 for x in pred_labels]
        # 遍历所有token
        for pred_label, token_id, t_type_id, t_offset in \
                zip(pred_labels, token_ids, token_type_ids, token_offsets):
            # 判断当前token是问题还是文章
            if 1 == t_type_id:
                # 若被预测为1，则加入答案关键字id集合
                if 1 == pred_label:
                    pred_answer_token_list.append((t_offset[0], t_offset[1], token_id))

        id_pred_token_dict[feature_recipe_id] = id_pred_token_dict.get(feature_recipe_id, []) + pred_answer_token_list

        # todo 并行加速，暂时不用，担心稳定性问题
        # infer_df = pd.DataFrame({
        #     'label': labels,
        #     'logit': [x for x in logits],
        #     'pred_label': pred_labels,
        #     'token_id': token_ids,
        #     't_type_id': token_type_ids,
        #     't_offset': token_offsets,
        # })
        # infer_df['pred_answer_token'] = infer_df.apply(
        #     lambda r: (r['t_offset'][0], r['t_offset'][1], r['token_id'])
        #     if (1 == r['t_type_id']) and (1 == r['pred_label'])
        #     else None, axis=1)

        # token的cross_entropy
        if label is not None:
            # 映射样本序列的label至特征序列
            labels = tag_offset_mapping(id_offset_maping_dict[feature_recipe_id], id_label_dict[feature_recipe_id],
                                        token_offsets, token_type_ids, 1, 0)
            logits_tensor = torch.tensor(logits, device="cuda" if torch.cuda.is_available() else "cpu").contiguous()
            labels_tensor = torch.tensor(labels, device="cuda" if torch.cuda.is_available() else "cpu").contiguous()
            single_loss = cross_entropy_fct(logits_tensor, labels_tensor)
            loss_list.append(single_loss.item())

    loss = sum(loss_list) / len(loss_list) if len(loss_list) > 0 else None

    # assert id_answer_dict.keys() == id_pred_token_dict.keys()

    # 将预测答案关键字整合成预测答案
    pred_result = []
    for recipe_id, pred_answer_token_list in id_pred_token_dict.items():
        pred_token_ids = sorted(list(set(pred_answer_token_list)), key=lambda x: x[0])
        pred_tokens = tokenizer.convert_ids_to_tokens([x[2] for x in pred_token_ids])
        pred_answer_middle = ' '.join(pred_tokens).replace(' ##', '').split(' ')
        pred_answer = ' '.join(set(pred_answer_middle))
        if '' == pred_answer:
            pred_answer = 'N/A'
        pred_result.append({
            'id': recipe_id,
            'answer': id_answer_dict[recipe_id],
            'pred_answer': pred_answer})

    return pred_result, loss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    upos_size: int = field(
        default=None,
        metadata={"help": "universal pos type size."},
    )
    entity_size: int = field(
        default=None,
        metadata={"help": "cook entity type size."},
    )
    embed_at_first_or_last: str = field(
        default=None, metadata={"help": "place the upos/entity embedding at first/last of the model."}
    )
    optimizer_type: str = field(
        default="AdamW",
        metadata={"help": "optimizer_type"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    use_upos: bool = field(
        default=True,
        metadata={"help": "use universal pos knowledge info representation"},
    )
    use_entity: bool = field(
        default=True,
        metadata={"help": "use cook entity knowledge info representation"},
    )

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def extract_qa_manager(raw_datasets):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    setattr(TrainingArguments, 'optimizer_type', 'AdamW')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.__setattr__('model_name_for_optimizer', os.path.split(model_args.model_name_or_path)[1].split('-')[0])
    training_args.__setattr__('optimizer_type', model_args.optimizer_type)

    # 获取upos和entity的词表长度信息
    if model_args.upos_size is None:
        model_args.upos_size = len(upos_map)
    if model_args.entity_size is None:
        model_args.entity_size = len(entity_map)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # 将upos_size、entity_size、embed_at_first_or_last参数配置到模型config中
    config.__setattr__('upos_size', model_args.upos_size)
    config.__setattr__('entity_size', model_args.entity_size)
    config.__setattr__('embed_at_first_or_last', model_args.embed_at_first_or_last)
    training_args.__setattr__('hidden_size', config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # model = AutoModelForQuestionAnswering.from_pretrained
    model = BertForExtractQA.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # todo
    # optimizer = utils.make_optimizer(training_args, model)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    upos_column_name = "upos" if "upos" in column_names else column_names[2]
    entity_column_name = "entity" if "entity" in column_names else column_names[3]
    offset_maping_column_name = "offset_maping" if "offset_maping" in column_names else column_names[4]
    label_column_name = "label" if "label" in column_names else column_names[5]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # original dataset cpreprocessing
    def prepare_features(examples, mode):

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # 一条原始样本，由于长度过长，可能被分割为多个训练样本
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # 分词后的token，在原始文本中的offset
        offset_mapping = tokenized_examples.pop("offset_mapping")
        if False:
            print(offset_mapping[0])
            print(examples['context'][0])
            # print(tokenized_examples['input_ids'][0])
            print(tokenizer.convert_ids_to_tokens(tokenized_examples['input_ids'][0]))

        # 样本的id名，由菜谱ID、问题序列号、问题，拼接而成
        tokenized_examples["recipe_id"] = []
        # 分词后的token，在原始文本中的offset
        tokenized_examples["token_offset"] = []
        # upos id
        if data_args.use_upos:
            tokenized_examples["upos_ids"] = []
        # entity id
        if data_args.use_entity:
            tokenized_examples["entity_ids"] = []

        if 'train' == mode:
            # 抽取式QA的标注
            tokenized_examples["extract_label"] = []

        for i, offsets in enumerate(offset_mapping):
            tokenized_examples["token_offset"].append(offsets)
            # We will label impossible answers with the index of the CLS token.
            # 文本的token所对应的segment_id的值
            context_index = 1 if pad_on_right else 0
            # input_ids = tokenized_examples["input_ids"][i]

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # 用于标识句子序号id，类似segment_id，但是区别在于：[cls]和[sep]被置为None
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            # 标识当前训练样本，对应原始语料样本的第几条
            sample_index = sample_mapping[i]
            # 赋值recipe_id
            tokenized_examples['recipe_id'].append(examples['id'][sample_index])

            # if 'train' == mode:
            # 提取原始语料样本的token的offset，和原始label
            ori_offset_maping = examples[offset_maping_column_name][sample_index]
            ori_label = examples[label_column_name][sample_index]
            ori_upos = examples[upos_column_name][sample_index]
            ori_entity = examples[entity_column_name][sample_index]

            """将原始样本的token标注（label，upos，entity），映射到分词后的训练样本token"""
            if data_args.use_upos:
                new_upos = tag_offset_mapping(ori_offset_maping, ori_upos, offsets, sequence_ids, 1, 0)
                tokenized_examples["upos_ids"].append(new_upos)
            if data_args.use_entity:
                new_entity = tag_offset_mapping(ori_offset_maping, ori_entity, offsets, sequence_ids, 1, 0)
                tokenized_examples["entity_ids"].append(new_entity)
            if 'train' == mode:
                new_extract_label = tag_offset_mapping(ori_offset_maping, ori_label, offsets, sequence_ids, 1, 0)
                tokenized_examples["extract_label"].append(new_extract_label)

        return tokenized_examples

    # Training preprocessing
    def prepare_train_features(examples):
        return prepare_features(examples, mode='train')

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        # 根据“最大训练样本数量”，提取相应条数的语料（第一次筛选）
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        # train_dataset_bak = prepare_train_features(train_dataset)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # 根据“最大训练样本数量”，提取相应条数的输入样本（第二次筛选），因为同一条语料可能被分割为多条样本。
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Validation preprocessing
    def prepare_vali_features(examples):
        return prepare_features(examples, mode='eval')

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        # while True:
        #     eval_dataset_bak = prepare_vali_features(eval_examples)
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_vali_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_vali_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        pred_result, loss = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
            tokenizer=tokenizer,
        )
        return pred_result, loss

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")
    # metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    if False:
        a = [{'id': '123',
              'prediction_text': 'f c b a'}]
        b = [{'id': '123',
              'answers': {'text': ['a b c f', 'd e'],
                          'answer_start': [100, 100]}}]
        metric.compute(predictions=a, references=b)

    def compute_metrics(pred_result):
        predictions = []
        references = []
        for item in pred_result:
            idx = item['id']
            answer_text = item['answer']
            prediction_text = item['pred_answer']
            predictions.append({'id': idx, 'prediction_text': prediction_text})
            references.append({'id': idx, 'answers': {'text': [answer_text], 'answer_start': [0]}})
        return metric.compute(predictions=predictions, references=references)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    # todo
    optimizer = utils.make_optimizer(training_args, model)
    # trainer.optimizer = optimizer

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    pred_results = None
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        pred_results = trainer.predict(predict_dataset, predict_examples)
        # metrics = results.metrics

        # max_predict_samples = (
        #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        # )
        # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        #
        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

    return pred_results, training_args.output_dir


if __name__ == '__main__':
    from data_process import data_process
    from datasets import Dataset

    dataset_model_vali, dataset_rule_vali = data_process('vali')
    dataset_model_vali = {key: value[:2] for key, value in dataset_model_vali.items()}
    dataset_model_vali = Dataset.from_dict(dataset_model_vali)
    datasets_model = {'train': dataset_model_vali, 'validation': dataset_model_vali, 'test': dataset_model_vali}

    model_pred_result, output_dir = extract_qa_manager(datasets_model)

    print('END')
