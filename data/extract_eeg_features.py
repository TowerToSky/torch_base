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


def get_trial_de(new_trails):
    features = []
    for trial in new_trails:
        feature = sufe.compute_DE(trial)
        features.append(feature)
    return np.array(features)

print(datetime.strftime(datetime.now(), "%Y-%m-%d-%A %H:%M:%S"))
# eeg_trials = np.empty((0,0,0))
# labels = np.empty((0))
# de_features = np.empty((0,0,0))
eeg_trials = []  # 表明1个人的长度
eeg_ex_index = []
labels = []
de_features = []
subject_indexs = [0]
subject_lists = [
    i
    for i in range(1, 36)
    if i != 11 and i != 9 and i != 19 and i != 20 and i != 33 and i != 15
]  # 15号眼动数据有问题
fs = 256
win_len = 1
# subject_indexs = np.empty((0))
# loop = tqdm.tqdm(subject_lists, total=len(subject_lists),desc="Processing participant")
# tracemalloc.start()
for participant_no in subject_lists:

    eeg_trial_list = sudr.get_eeg_trials(participant_no, use_ICA=True)
    label_list = sudr.get_labels(participant_no)
    label_list = label_list[:, -1] - 1
    ex_index = [0]
    for ex_no in range(len(eeg_trial_list)):
        new_trails, new_labels = sudp.re_data_slide(
            trial=eeg_trial_list[ex_no],
            label=label_list[ex_no],
            win_len=win_len * fs,
            overlap=0,
            is_filter=False,
            norm_method=None,
        )  # 因为没有overlap，所以用1s大小更为通用一些
        de_feature = get_trial_de(new_trails)
        de_features.extend(de_feature)
        eeg_trials.extend(new_trails)
        labels.extend(new_labels)
        ex_index.append(len(new_trails) + ex_index[-1])

    subject_indexs.append(len(eeg_trials))
    eeg_ex_index.append(ex_index)
    # subject_indexs = np.append(subject_indexs, len(eeg_trials))
    print(
        "Processing participant %d, %s,%s,%s,%s"
        % (
            participant_no,
            str(len(eeg_trials)),
            str(len(de_features)),
            str(len(labels)),
            str(len(eeg_ex_index)),
        )
    )

    # print(len(eeg_trials),len(de_features),len(labels))

eeg_trials = np.asarray(eeg_trials)  # n,256,31

labels = np.asarray(labels)  # n
de_features = np.asarray(de_features)  # n,5,31
# de_features = de_features.swapaxes(1, 2)
subject_indexs = np.array(subject_indexs)  # n + 1
eeg_ex_index = np.array(eeg_ex_index)
print(
    eeg_trials.shape,
    labels.shape,
    de_features.shape,
    subject_indexs.shape,
    eeg_ex_index.shape,
)
print(subject_indexs)

# sys.exit(0)

# 保存数据
save_dir_path = f"/data/SpaticalTest/Spatial/total_data_{win_len}s_0"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
np.save(os.path.join(save_dir_path, f"eeg_trials_{win_len}s_0.npy"), eeg_trials)
np.save(os.path.join(save_dir_path, f"labels_{win_len}s_0.npy"), labels)
np.save(os.path.join(save_dir_path, f"de_features_{win_len}s_0.npy"), de_features)
np.save(os.path.join(save_dir_path, f"subject_index_{win_len}s_0.npy"), subject_indexs)
np.save(os.path.join(save_dir_path, f"eeg_ex_index_{win_len}s_0.npy"), eeg_ex_index)
print(
    "数据保存成功, 保存路径为: %s, %s, %s, %s, %s, %s"
    % (
        save_dir_path,
        f"eeg_trials_{win_len}s_0.npy",
        f"labels_{win_len}s_0.npy",
        f"de_features_{win_len}s_0.npy",
        f"subject_index_{win_len}s_0.npy",
        f"eeg_ex_index_{win_len}s_0.npy",
    )
)
