import torch
import numpy as np
import sys
import spatial_utils.data_process as sudp
import spatial_utils.data_read as sudr
import spatial_utils.feature_extract as sufe

# 好像不清楚滑窗截取的数据存在哪儿了，重新整一下
import tqdm
from datetime import datetime
import os
import pandas as pd
import time
import tracemalloc

eeg_trials = []
labels = []
de_features = []
subject_indexs = [0]
eeg_ex_index = []
subject_lists = [
    i for i in range(1, 36) if i != 11 and i != 9 and i != 19 and i != 20 and i != 33
]
print("2024-04-06 20:10")

for participant_no in subject_lists:
    print(participant_no)
    eeg_trial_list = sudr.get_eeg_trials(participant_no, use_ICA=True)
    label_list = sudr.get_labels(participant_no)
    label_list = label_list[:, -1] - 1
    per_trial_index = [0]
    eeg_trials = []
    for ex_no in range(len(eeg_trial_list)):
        new_trails, new_labels = sudp.re_data_slide(
            trial=eeg_trial_list[ex_no],
            label=label_list[ex_no],
            win_len=256,
            overlap=0,
            is_filter=False,
            norm_method=None,
        )  # 因为没有overlap，所以用1s大小更为通用一些
        eeg_trials.extend(new_trails)
        per_trial_index.append(len(eeg_trials))
    print(per_trial_index)
    eeg_ex_index.append(per_trial_index)

test = np.array(eeg_ex_index)
np.save("/data/SpaticalTest/Spatial/total_33_eeg_ex_index.npy", test)
