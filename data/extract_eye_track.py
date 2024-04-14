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
    i
    for i in range(1, 36)
    if i != 11 and i != 9 and i != 19 and i != 20 and i != 33 and i != 15
]

print(datetime.strftime(datetime.now(), "%Y-%m-%d-%A %H:%M:%S"))
win_len = 4
fs = 60

dir_path = f"/data/SpaticalTest/Spatial/total_data_{win_len}s_0"
path_end = f"_{win_len}s_0.npy"
eeg_ex_index = np.load(os.path.join(dir_path, ("eeg_ex_index" + path_end)))
subject_index = np.load(os.path.join(dir_path, ("subject_index" + path_end)))
total_eye_data = []
total_eye_label = []

# eeg_index = np.concatenate((eeg_index[:12],eeg_index[13:])) # 排除15号，4s的数据没有用15号

for ids, per_no in enumerate(subject_lists):
    print(f"Processing {per_no} ...")
    per_eye_data = []
    per_eye_label = []
    eye_data = sudr.get_eye_track_trials_4(per_no)
    eye_labels = sudr.get_labels(per_no)
    eye_labels = eye_labels[:, -1] - 1
    for i, (data, label) in enumerate(zip(eye_data, eye_labels)):
        new_data, new_label = sudp.re_data_slide(
            data,
            label,
            win_len=fs * win_len,
            overlap=0,
            is_filter=None,
            norm_method=None,
        )
        # 均值补齐
        eeg_len = eeg_ex_index[ids][i + 1] - eeg_ex_index[ids][i]
        diff = eeg_len - len(new_data)
        if diff > 0:
            new_data = np.pad(new_data, ((0, diff), (0, 0), (0, 0)), mode="mean")
        elif diff < 0:
            new_data = new_data[:diff]
            eeg_len = len(new_data)

        new_label = np.array([label] * eeg_len)
        # print(i, len(new_data), len(new_label))
        per_eye_data.append(new_data)
        per_eye_label.append(new_label)
    per_eye_data = np.concatenate(per_eye_data)
    per_eye_label = np.concatenate(per_eye_label)
    print(f"{per_no} per_eye_data.shape: {per_eye_data.shape}")
    print(f"{per_no} per_eye_label.shape: {per_eye_label.shape}")
    print(f"{per_no} eeg_data_len: {subject_index[ids+1] - subject_index[ids]}")
    total_eye_data.append(per_eye_data)
    total_eye_label.append(per_eye_label)

total_eye_data = np.concatenate(total_eye_data)
total_eye_label = np.concatenate(total_eye_label)
print(total_eye_data.shape)
print(total_eye_label.shape)

np.save(os.path.join(dir_path, ("eye_track" + path_end)), total_eye_data)
np.save(os.path.join(dir_path, ("eye_labels" + path_end)), total_eye_label)
print("Done!")
