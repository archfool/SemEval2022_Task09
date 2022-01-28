# -*- coding: utf-8 -*-
"""
Created on 2022/1/28 17:06
author: ruanzhihao_archfool
"""

# from utils.lamb import Lamb
# from madgrad import MADGRAD
from transformers import AdamW
from torch.optim import Adam


def get_optimizer_params_large(args, model):
    # Grouped Layer-wise Learning Rate Decay (GLLRD) FOR LARGE MODEL

    no_decay = ['bias', 'gamma', 'beta']
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    group2 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.', 'layer.12.', 'layer.13.', 'layer.14.', 'layer.15.']
    group3 = ['layer.16.', 'layer.17.', 'layer.18.', 'layer.19.', 'layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                 'layer.8.', 'layer.9.',
                 'layer.10.', 'layer.11.', 'layer.12.', 'layer.13.', 'layer.14.', 'layer.15.', 'layer.16.', 'layer.17.',
                 'layer.18.', 'layer.19.', 'layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']

    optimizer_grouped_parameters = [
        # 首先对所有层中需要权重衰减的need_decay参数设置weight_decay
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        # 对group1里的所有需要权重衰减的need_decay参数设置较低的学习率learning_rate/2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate / 1.6},
        # 对group2里所有需要权重衰减need_decay的参数设置中等学习率learning_rate
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        # 对group3里所有的需要权重衰减的参数设置较高的学习率learning_rate * 2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate * 1.6},
        # 首先对所有层中不需要权重衰减的no_decay参数设置0的weight_decay
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        # 对group1里的所有不需要权重衰减的no_decay参数设置较低的学习率learning_rate/2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.learning_rate / 1.6},
        # 对group2里的所有不需要权重衰减的no_decay参数设置中等的学习率learning_rate
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        # 对group3里的所有不需要权重衰减的no_decay参数设置较高的学习率learning_rate * 2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.learning_rate * 1.6},
        # 对非transformer结构的Head层设置极高的学习率
        {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': args.learning_rate * 4,
         "momentum": 0.99},
        # args.learning_rate*2
    ]
    return optimizer_grouped_parameters


def get_optimizer_params_base(args, model):
    """
    For Base Model
    """
    # 将参数分为需要权重衰减的need_decay参数和不需要权重衰减的no_decay参数
    no_decay = ['bias', 'gamma', 'beta']
    # 将所有的预训练层分成3组
    group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
    group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
    group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
    group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']

    name_optimizer_grouped_parameters = [
        # 首先对所有层中需要权重衰减的need_decay参数设置weight_decay
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        # 对group1里的所有需要权重衰减的need_decay参数设置较低的学习率learning_rate/2.6
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate / 1.6},
        # 对group2里所有需要权重衰减need_decay的参数设置中等学习率learning_rate
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        # 对group3里所有的需要权重衰减的参数设置较高的学习率learning_rate * 2.6
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate * 1.6},
        # 首先对所有层中不需要权重衰减的no_decay参数设置0的weight_decay
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        # 对group1里的所有不需要权重衰减的no_decay参数设置较低的学习率learning_rate/2.6
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.learning_rate / 1.6},
        # 对group2里的所有不需要权重衰减的no_decay参数设置中等的学习率learning_rate
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        # 对group3里的所有不需要权重衰减的no_decay参数设置较高的学习率learning_rate * 2.6
        {'params': [n for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.learning_rate * 1.6},
        # 对非transformer结构的Head层设置极高的学习率
        {'params': [n for n, p in model.named_parameters() if "bert" not in n], 'lr': args.learning_rate * 5,
         "momentum": 0.99},
    ]

    optimizer_grouped_parameters = [
        # 首先对所有层中需要权重衰减的need_decay参数设置weight_decay
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        # 对group1里的所有需要权重衰减的need_decay参数设置较低的学习率learning_rate/2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate / 1.6},
        # 对group2里所有需要权重衰减need_decay的参数设置中等学习率learning_rate
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        # 对group3里所有的需要权重衰减的参数设置较高的学习率learning_rate * 2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate * 1.6},
        # 首先对所有层中不需要权重衰减的no_decay参数设置0的weight_decay
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        # 对group1里的所有不需要权重衰减的no_decay参数设置较低的学习率learning_rate/2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.learning_rate / 1.6},
        # 对group2里的所有不需要权重衰减的no_decay参数设置中等的学习率learning_rate
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        # 对group3里的所有不需要权重衰减的no_decay参数设置较高的学习率learning_rate * 2.6
        {'params': [p for n, p in getattr(model, args.model_name_for_optimizer).named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.learning_rate * 1.6},
        # 对非transformer结构的Head层设置极高的学习率
        {'params': [p for n, p in model.named_parameters() if "bert" not in n], 'lr': args.learning_rate * 5,
         "momentum": 0.99},
    ]

    return optimizer_grouped_parameters


def make_optimizer(args, model):
    # 定义optimizer和各个层的学习率等参数
    optimizer_name = args.optimizer_type
    if args.hidden_size == 768:
        optimizer_grouped_parameters = get_optimizer_params_base(args, model)
    elif args.hidden_size == 1024:
        optimizer_grouped_parameters = get_optimizer_params_large(args, model)
    else:
        raise Exception('args.hidden_size unacceptable!')

    kwargs_1 = {
        'betas': (0.9, 0.98),
        "weight_decay": args.weight_decay,
        'lr': args.learning_rate,
        'eps': args.adam_epsilon,
        'correct_bias': True
    }
    kwargs_2 = {
        "weight_decay": args.weight_decay,
        'lr': args.learning_rate,
        'betas': (0.9, 0.98),
        'eps': args.adam_epsilon,
        'correct_bias': True  # not args.use_bertadam
    }
    # kwargs_3 = {
    #     #   'lr':args.learning_rate, # lr: float = 1e-2
    #     'weight_decay': args.weight_decay,  # weight_decay: float = 0
    #     'eps': args.epsilon,  # eps: float = 1e-6
    #     'momentum': 0.9,  # momentum: float = 0.9
    # }

    if optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs_2)
        return optimizer
    elif optimizer_name == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, **kwargs_1)
        return optimizer
    # elif optimizer_name == "LAMB":
    #     optimizer = Lamb(optimizer_grouped_parameters, **kwargs_1)
    #     return optimizer
    # elif optimizer_name == "MADGRAD":
    #     # lr: float = 1e-2, momentum: float = 0.9,
    #     # weight_decay: float = 0, eps: float = 1e-6,
    #     optimizer = MADGRAD(optimizer_grouped_parameters, **kwargs_3)
    #     return optimizer
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))
