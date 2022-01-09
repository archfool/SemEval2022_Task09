# -*- coding: utf-8 -*-
"""
Created on 2022/1/9 21:40
author: ruanzhihao_archfool
"""

import os
import sys
import pandas as pd

from data_process import data_process

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from manager import manager
from util_tools import logger
from util_model import ACT2FN


if __name__ == "__main__":
    corpus = data_process()
    print('END')
