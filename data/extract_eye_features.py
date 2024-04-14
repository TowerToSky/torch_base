import pandas as pd
import os
import numpy
from spatial_utils.eye_utils import *
from tqdm import tqdm
from datetime import datetime

def get_trial(subject_no):
    """
    Args:
        subject_no：受试者ID
    Returns:
        sub_trials_list：受试者在做1-54道题目时的眼动数据，包含瞳孔直径（左，右）、注视点（X，Y）、眼动类型、注视时间
    """
    sub_trials_list = []

    for stimulus_id in range(1, 55):
        print("正在读取第%d受试者的第%d道题目的眼动数据" % (subject_no, stimulus_id))
        file_path = r"/data/SpaticalTest/EyeTrackData/subject%02d/exercise%02d.xlsx" % (
            subject_no,
            stimulus_id,
        )
        df = pd.read_excel(file_path)
        # 把列名中的空格都换成下划线
        name_map = {col: col.replace(" ", "_") for col in df.columns}
        df = df.rename(columns=name_map)

        df = df[1 : len(df) - 1]  # 去掉第一行和最后一行

        # 选取键盘事件之前的数据
        # TODO:存疑
        # if('KeyboardEvent' in df.Event.values):
        #     key_dex = df[df['Event'] == 'KeyboardEvent'].index.tolist()
        #     df = df.loc[:key_dex[-1]-1]

        # 去除眼动类型为未分类的数据
        un_dex = df.index[df["Eye_movement_type"] == "Unclassified"].tolist()
        df.drop(index=un_dex, axis=0, inplace=True)

        # 删type列空值
        df.dropna(axis=0, subset=["Eye_movement_type"])

        # 插值
        df[(df["Validity_right"] == "Invalid") | (df["Validity_left"] == "Invalid")] = (
            df.interpolate()
        )

        df = df.interpolate(limit_direction="both")

        df = df.loc[
            :,
            [
                "Pupil_diameter_left",
                "Pupil_diameter_right",
                "Gaze_point_X",
                "Gaze_point_Y",
                #             "Fixation_point_X","Fixation_point_Y",
                "Eye_movement_type",
                "Gaze_event_duration",
            ],
        ]

        # 将眼动的三种类型修改为0,1,2
        new = {"EyesNotFound": 0, "Fixation": 1, "Saccade": 2}
        df["Eye_movement_type"] = df["Eye_movement_type"].map(new)

        df = df.reset_index(drop=True)  # 重新编号
        sub_trials_list.append(df.values.tolist())

    return sub_trials_list


def getEyeTrackFeatures(subject_no, fs=60, win_len=4, overlap=0):
    """
    Args:
        subject_no：受试者ID
    Returns:
        all_trials_list：受试者在做1-54道题目时的眼动数据，以win_len=60，overlap=0.25滑窗截取，
                         对于每一个滑窗提取眼动特征包含
                            blink_count, avg_blink_duration, std_blink_duration, min_blink_duration, max_blink_duration,
                            fixation_count, avg_fixation_duration, std_fixation_duration, min_fixation_duration, max_fixation_duration, avg_x, std_x, skew_x, kurt_x, avg_y, std_y, skew_y, kurt_y, avg_disper, std_disper, max_disper,
                            saccade_count, avg_saccade_duration, std_saccade_duration, min_saccade_duration, max_saccade_duration, avg_saccade_amplitude, std_saccade_amplitude, avg_saccade_velocity,
                            avg_lpupil, std_lpupil, min_lpupil, max_lpupil, skew_lpupil, kurt_lpupil, avg_rpupil, std_rpupil, min_rpupil, max_rpupil, skew_rpupil, kurt_rpupil
                         共计41个特征
                         返回一个np.array类型，即一个受试者在实验过程中产生的眼动特征
                         2023-12-4日：根据图神经网络输入类型，为了匹配脑电输入形状，眼动特征必须为(54,*)的形状

    """
    all_trials_list = []
    # try:
    eyetrack_trials = get_trial(
        subject_no
    )  # 获取到的数据是根据每道题目划分好的，也就是54个list，这个很慢的原因是读xlxs文件，很烦，先不改了，先去读然后分

    for i, eyetrack_trial in enumerate(eyetrack_trials):

        # print(" 第%d个excel"%(i+1))
        # 滑窗截取数据
        new_trials, shape = slide_trial(
            np.array(eyetrack_trial),
            wid_len=fs * win_len,
            overlap=overlap,
            is_slide=True,
        )
        # print(" 滑窗形状：",shape)
        per_win_features = []
        for pid in range(shape[0]):
            # 根据滑窗堆叠其他数据
            # print("  第%d个滑窗"%(pid))
            # 构建1s内的df
            df_trial = pd.DataFrame(
                new_trials[pid],
                columns=[
                    "Pupil_diameter_left",
                    "Pupil_diameter_right",
                    "Gaze_point_X",
                    "Gaze_point_Y",
                    "Eye_movement_type",
                    "Gaze_event_duration",
                ],
            )
            # print(df_trial)

            # blink_count, avg_blink_duration, std_blink_duration, min_blink_duration, max_blink_duration
            blinkParams = findBlinkParams(df_trial)

            # fixation_count, avg_fixation_duration, std_fixation_duration, min_fixation_duration, max_fixation_duration, avg_x, std_x, skew_x, kurt_x, avg_y, std_y, skew_y, kurt_y,avg_disper, std_disper, max_disper
            fixationParams = findFixationParams(df_trial)

            # saccade_count, avg_saccade_duration, std_saccade_duration, min_saccade_duration, max_saccade_duration, avg_saccade_amplitude, std_saccade_amplitude, avg_saccade_velocity
            saccadeParams = findSaccadeParams(df_trial)

            # avg_lpupil, std_lpupil, min_lpupil, max_lpupil, skew_lpupil, kurt_lpupil, avg_rpupil, std_rpupil, min_rpupil, max_rpupil, skew_rpupil, kurt_rpupil
            pupilParams = findPupilParams(df_trial)

            eyeTrackFeatures = np.hstack(
                blinkParams + fixationParams + saccadeParams + pupilParams
            )
            # print("  eyeTrackFeatures:",eyeTrackFeatures.shape)
            per_win_features.append(eyeTrackFeatures)

        per_win_features = np.array(per_win_features)

        # 这里需要改一下，将一个trial的数据堆叠
        all_trials_list.append(per_win_features)
    # except:
    #     print("第%d受试者眼动数据提取特征时出现问题"%subject_no)

    return np.array(all_trials_list, dtype=object)


if __name__ == "__main__":
    # subject_lists = [33]
    print(datetime.strftime(datetime.now(), "%Y-%m-%d-%A %H:%M:%S"))
    subject_lists = [
        i
        for i in range(1, 36)
        if i != 19 and i != 20 and i != 9 and i != 33 and i != 15
    ]
    win_len = 1
    fs = 60
    print(subject_lists)
    path = f"/data/SpaticalTest/Spatial/eye_features_{win_len}"
    os.makedirs(path, exist_ok=True)
    with tqdm(range(len(subject_lists)), desc="Processing", unit="iteration") as t:
        for subject_no in subject_lists:
            t.update(1)
            print("正在提取第%02d号受试者的眼动数据" % subject_no)
            eyeTrackFeatures = getEyeTrackFeatures(subject_no, fs=fs, win_len=win_len)
            file_path = os.path.join(path, f"{subject_no}.npy")
            print("eyeTrackFeatures.shape：", eyeTrackFeatures.shape)
            np.save(file_path, eyeTrackFeatures)
