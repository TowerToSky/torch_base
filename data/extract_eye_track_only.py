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

# 只是为了提取眼动数据，不需要与脑电数据对齐，提取每个ex_index，封装成每个trial
subject_lists = [
    i
    for i in range(1, 36)
    if i != 11 and i != 9 and i != 19 and i != 20 and i != 33 and i != 15
]

print(datetime.strftime(datetime.now(), "%Y-%m-%d-%A %H:%M:%S"))
win_len = 1
fs = 60

total_eye_data = []
total_eye_label = []
eye_track_data = []
eye_ex_index = []
eye_subject_indexs = [0]

# eeg_index = np.concatenate((eeg_index[:12],eeg_index[13:])) # 排除15号，4s的数据没有用15号

for ids, per_no in enumerate(subject_lists):  # 每个人
    print(f"Processing {per_no} ...")
    eye_data = sudr.get_eye_track_trials_4(per_no)
    eye_labels = sudr.get_labels(per_no)
    eye_labels = eye_labels[:, -1] - 1
    eye_index = [0]
    for i, (data, label) in enumerate(zip(eye_data, eye_labels)):  # 每个trial
        new_data, new_label = sudp.re_data_slide(
            data,
            label,
            win_len=fs * win_len,
            overlap=0,
            is_filter=None,
            norm_method=None,
        )  # 滑窗构成的每win_len数据
        total_eye_data.extend(new_data)
        total_eye_label.extend(new_label)
        eye_index.append(len(new_data) + eye_index[-1])
    eye_ex_index.append(eye_index)
    eye_subject_indexs.append(len(total_eye_data))

    print(
        "Processing participant %d, %s,%s,%s,%s"
        % (
            per_no,
            str(len(total_eye_data)),
            str(len(total_eye_label)),
            str(len(eye_ex_index)),
            str(len(eye_subject_indexs)),
        )
    )
    print(eye_index)
    print(eye_subject_indexs)

total_eye_data = np.array(total_eye_data)
total_eye_label = np.array(total_eye_label)
eye_ex_index = np.array(eye_ex_index)
eye_subject_indexs = np.array(eye_subject_indexs)
print(
    total_eye_data.shape,
    total_eye_label.shape,
    eye_ex_index.shape,
    eye_subject_indexs.shape,
)
print(eye_subject_indexs)

# 保存数据
save_dir_path = f"/data/SpaticalTest/Spatial/total_data_{win_len}s_0"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
np.save(os.path.join(save_dir_path, f"eye_track_{win_len}s_0.npy"), total_eye_data)
np.save(os.path.join(save_dir_path, f"eye_labels_{win_len}s_0.npy"), total_eye_label)
np.save(
    os.path.join(save_dir_path, f"eye_subject_index_{win_len}s_0.npy"),
    eye_subject_indexs,
)
np.save(os.path.join(save_dir_path, f"eye_ex_index_{win_len}s_0.npy"), eye_ex_index)
print(
    "数据保存成功, 保存路径为: %s, %s, %s, %s, %s"
    % (
        save_dir_path,
        f"eye_track_{win_len}s_0.npy",
        f"eye_labels_{win_len}s_0.npy",
        f"eye_subject_index_{win_len}s_0.npy",
        f"eye_ex_index_{win_len}s_0.npy",
    )
)
