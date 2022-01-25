import copy
import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm

tags = {'Lead': 0,'B-Lead': 0,'I-Lead': 7,'Position': 1,'B-Position': 2,'I-Position': 8,'Evidence': 2,'B-Evidence': 4,
        'I-Evidence': 9,'Claim': 3,'B-Claim': 6,'I-Claim': 10,'Concluding Statement': 4,'B-Concluding Statement': 8,
        'I-Concluding Statement': 11,'Counterclaim': 5,'B-Counterclaim': 10,'I-Counterclaim': 12,'Rebuttal': 6,'B-Rebuttal': 12,
        'I-Rebuttal': 13,'O': 14,'Special': -100}

def _prepare_training_data_helper(args,token,targets):
    train_sample = []
    for i in range(len(token)):
        sample = {}
        sample['input_ids'] = token[i]
        sample['targets'] = targets[i]
        train_sample.append(sample)
    return train_sample

def prepare_training_data(args,token,targets):
    return _prepare_training_data_helper(args,token,targets)
    