import os
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


## 根据受试者编号读取其脑电数据
def get_eeg_trials(subject_id, use_ICA=False):
    """
    Introduction:
        根据受试者编号获取受试者所有trial
    Args:
        subject_id: 受试者编号
        return: eeg_trials
    """
    if subject_id <= 18:
        path = "/data/SpaticalTest/SpaticalAbilityTest/old/subject%02d/eeg/%06d.vhdr"  # 脑电数据的头文件，记录一些元信息，一些采集的参数
        path_vmrk = "/data/SpaticalTest/SpaticalAbilityTest/old/subject%02d/eeg/%06d.vmrk"  # 脑电数据的标记文件，记录trigger信息
    else:
        path = "/data/SpaticalTest/SpaticalAbilityTest/202307/%03d/eeg/%03d.vhdr"  # 脑电数据的头文件，记录一些元信息，一些采集的参数
        path_vmrk = "/data/SpaticalTest/SpaticalAbilityTest/202307/%03d/eeg/%03d.vmrk"  # 脑电数据的标记文件，记录trigger信息

    raw = mne.io.read_raw_brainvision(
        path % (subject_id, subject_id), verbose=False, preload=True
    )  # 会自动关联.eeg,.vhdr,.vmrk文件
    ## 滤波、陷波、重采样
    raw_filter = raw.filter(0.5, 70, fir_design="firwin", verbose=False)
    raw_filter = raw_filter.notch_filter(50, verbose=False)
    raw_filter.resample(256, npad="auto", verbose=False)

    if use_ICA == True:
        ica = ICA(n_components=31, random_state=97, verbose=False)
        ica.fit(raw_filter, verbose=False)
        eog_indices, eog_scores = ica.find_bads_eog(
            raw_filter, ch_name=["Fp1", "Fp2"], verbose=False
        )
        eog_scores_abs = sum(abs(np.array(eog_scores))).tolist()
        ica.exclude = [eog_scores_abs.index(max(eog_scores_abs))]
        ica.apply(raw_filter, verbose=False)

    annot = mne.read_annotations(path_vmrk % (subject_id, subject_id))
    raw_filter.set_annotations(annot)
    events, event_ids = mne.events_from_annotations(raw_filter)

    ## 根据trigger分段数据，trigger1是指导语；trigger2是冷静的风景画面（12张图，12个trigger2）；trigger3是注视点；
    # trigger4是第一次测试程序（空间推理30道题，30个trigger4）；trigger5是第二次测试程序（空间旋转24道题，24个trigger5）；trigger6是结束
    ## 每个trial是从一个注视点出现到下一个注视点出现这段数据
    inference_start_time = events[:, 0][np.where(events[:, 2] == 3)[0]][0]
    rotation_start_time = events[:, 0][np.where(events[:, 2] == 3)[0]][30]
    inference_index = np.where(events[:, 0] == inference_start_time)[0][0]
    rotation_index = np.where(events[:, 0] == rotation_start_time)[0][0]
    inference_trail_index = events[:, 0][inference_index:rotation_index]
    rotation_trail_index = events[:, 0][rotation_index:]

    ## 读取数据
    inference_trail_list = []
    rotation_trail_list = []
    raw_filter_data = raw_filter.get_data()
    for i in np.arange(1, len(inference_trail_index), 2):
        trial = raw_filter_data[
            :, inference_trail_index[i] : inference_trail_index[i + 1]
        ].T
        inference_trail_list.append(trial)
    for i in np.arange(1, len(rotation_trail_index), 2):
        trial = raw_filter_data[
            :, rotation_trail_index[i] : rotation_trail_index[i + 1]
        ].T
        rotation_trail_list.append(trial)
    del raw_filter_data
    trial_list = inference_trail_list + rotation_trail_list
    return trial_list


def get_labels(subject_id):
    """
    Introduction:
        根据受试者编号获取受试者所有trial的标签
    Args:
        subject_id: 受试者编号
        return: subject_labels,有三列，分别是答题正确与否，是否困惑，维度，根据实际情况选择
    """
    if subject_id <= 18:
        label_path = "/data/SpaticalTest/SpaticalAbilityTest/old/class_all.xlsx"
        sheet_name = "被试%d" % (subject_id)
    else:
        label_path = "/data/SpaticalTest/SpaticalAbilityTest/202307/class_all.xlsx"
        sheet_name = "%d" % (subject_id)

    class_all = pd.read_excel(label_path, sheet_name=sheet_name)

    # 54 * 5,其中第5列没用，第一列表示题号，第二列表示答题正确与否，第三列表示是否困惑，第四列表示困惑维度（Confused,Think-right,Guess,Non-confused）
    # 题目1-30为空间推理测试,30-54为空间旋转测试
    # 将class_all中第2列至第4列数据提取出来
    labels = class_all.iloc[:, 1:4]
    # 将第三列数据映射成数字
    labels[labels.columns[2]] = labels.iloc[:, 2].map(
        {"Confused": 1, "Guess": 2, "Non-confused": 3, "Think-right": 4}
    )
    labels = np.asarray(labels)
    return labels


def get_eye_track_trials(subject_id):
    """
    Introduction:
        根据受试者编号读取其眼动数据
    Args:
        subject_id: 受试者编号
        return: eye_track_trials
    """
    if subject_id <= 18:
        eye_track_path = "/data/SpaticalTest/eye_data/subject%02d.xlsx" % subject_id
    else:
        path = "/data/SpaticalTest/SpaticalAbilityTest/202307/%03d/eye/" % subject_id
        eye_track_path = ""
        for file_name in os.listdir(path):
            if file_name.endswith(".xlsx"):
                eye_track_path = os.path.join(path, file_name)
    # path = "/data/Ruiwen/原始数据/瑞文眼动数据（20210518）/experiment_%d.xlsx"
    data = pd.read_excel(eye_track_path)

    ## 左右瞳孔直径大小数据
    pupil_data = data[["Pupil diameter left", "Pupil diameter right"]]

    event_value = data[["Event value"]].dropna(axis=0, how="all")
    index1 = event_value[event_value["Event value"] == "fixation.jpg"].index
    index2 = data[data["Event"] == "RecordingEnd"].index
    trial_index = np.concatenate([index1, index2])

    trial_list = []
    for i in np.arange(1, len(trial_index), 2):
        trial = pupil_data.loc[trial_index[i] : trial_index[i + 1]].dropna(
            axis=0, how="all"
        )
        trial = np.asarray(trial)
        ## 使用-1填充缺失值
        trial[np.isnan(trial)] = -1
        trial_list.append(np.asarray(trial))
    return trial_list


def get_eye_track_trials_4(subject_id):
    """
    Introduction:
        根据受试者编号读取其眼动数据
    Args:
        subject_id: 受试者编号
        return: eye_track_trials
    """
    if subject_id <= 18:
        eye_track_path = "/data/SpaticalTest/eye_data/subject%02d.xlsx" % subject_id
    else:
        path = "/data/SpaticalTest/SpaticalAbilityTest/202307/%03d/eye/" % subject_id
        eye_track_path = ""
        for file_name in os.listdir(path):
            if file_name.endswith(".xlsx"):
                eye_track_path = os.path.join(path, file_name)
    # path = "/data/Ruiwen/原始数据/瑞文眼动数据（20210518）/experiment_%d.xlsx"
    data = pd.read_excel(eye_track_path)

    eye_data = data[
        [
            "Pupil diameter left",
            "Pupil diameter right",
            "Eye movement type",
            "Gaze event duration",
        ]
    ]
    event_value = data[["Event value"]].dropna(axis=0, how="all")
    index1 = event_value[event_value["Event value"] == "fixation.jpg"].index
    index2 = data[data["Event"] == "RecordingEnd"].index
    trial_index = np.concatenate([index1, index2])
    trial_list = []

    for i in np.arange(1, len(trial_index), 2):
        # trial_index = eye_data['Pupil diameter left'].loc[trial_index[i]:trial_index[i+1]].index
        trial = eye_data.loc[
            trial_index[i] : trial_index[i + 1]
        ]  # 提取一个实验trial里的眼动数据
        # 对于只有一只眼睛的瞳孔直径的情况，直接补齐处理
        trial.loc[trial["Pupil diameter right"].isna(), "Pupil diameter right"] = trial[
            "Pupil diameter left"
        ]
        trial.loc[trial["Pupil diameter left"].isna(), "Pupil diameter left"] = trial[
            "Pupil diameter right"
        ]
        # trial = trial.dropna(axis=0, how="any")
        # trial = eye_data.iloc[trial_index].copy()
        trial.loc[trial["Eye movement type"] != "Fixation", "Eye movement type"] = 0
        trial.loc[trial["Eye movement type"] == "Fixation", "Eye movement type"] = 1
        trial = np.asarray(trial, dtype=np.float32)
        ## 使用-1填充缺失值
        trial[np.isnan(trial)] = -1
        trial_list.append(
            np.asarray(trial)
        )  # 一个人做了54道题，每道题采集了不同时长的眼动数据

    return trial_list


def get_eye_track_trials_my(subject_id):

    path = "/data/Ruiwen/原始数据/瑞文眼动数据（20210518）/experiment_%d.xlsx"
    eye_track_path = path % (subject_id)
    data = pd.read_excel(eye_track_path)

    eye_data = data[
        [
            "Pupil diameter left",
            "Pupil diameter right",
            "Eye movement type",
            "Gaze event duration",
        ]
    ]
    event_value = data[["Event value"]].dropna(axis=0, how="all")
    index1 = event_value[event_value["Event value"] == "fixation.jpg"].index
    index2 = data[data["Event"] == "RecordingEnd"].index
    trial_index = np.concatenate([index1, index2])
    trial_list = []

    for i in np.arange(1, len(trial_index), 2):
        # trial_index = eye_data['Pupil diameter left'].loc[trial_index[i]:trial_index[i+1]].index
        trial = eye_data.loc[
            trial_index[i] : trial_index[i + 1]
        ]  # 提取一个实验trial里的眼动数据
        trial = trial.dropna(axis=0, how="any")
        # trial = eye_data.iloc[trial_index].copy()
        trial.loc[trial["Eye movement type"] != "Fixation", "Eye movement type"] = 0
        trial.loc[trial["Eye movement type"] == "Fixation", "Eye movement type"] = 1
        trial = np.asarray(trial, dtype=np.float32)
        ## 使用-1填充缺失值
        trial[np.isnan(trial)] = -1
        trial_list.append(
            np.asarray(trial)
        )  # 一个人做了48道题，每道题采集了不同时长的眼动数据

    return trial_list


def get_subject_data(subject_id):
    """
    Introduction:
        根据受试者编号读取其eeg、眼动还有标签数据
    Args:
        subject_id: 受试者编号
        return: eeg_trials, eye_track_trials, subject_labels
    """

    return (
        get_eeg_trials(subject_id),
        get_eye_track_trials(subject_id),
        get_labels(subject_id),
    )
    # return get_eeg_trials(subject_id), get_eye_track_trials_my(subject_id), get_labels(subject_id)
