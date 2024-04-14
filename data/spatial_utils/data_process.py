from scipy import signal
from scipy.fftpack import fft, ifft
from sklearn import preprocessing
import numpy as np
import random
from scipy.signal import butter, filtfilt

def filter_data(low, high, data, fs=256):
    """
    Introduction:
        带通滤波
    Args:
        data shape: sample * channel
        low: 滤波最低通过频率
        high: 滤波最高通过频率
        fs: 原始数据采样率
        return: 滤波处理后的数据
    """    
    b, a = signal.butter(4, [2*low/fs, 2*high/fs], 'bandpass')
    filter_data = []
    for chan_index in np.arange(data.shape[1]):
        # print(chan_index)
        data_f = signal.filtfilt(b, a, data[:,chan_index])
        filter_data.append(data_f)
        
    return np.asarray(filter_data).T

def butterworth_filter(data_raw, fs, lcf=1, hcf=70, order=4):
    """
    Introduction:
        n阶巴特沃斯带通滤波器
    Args:
        data: 二维数组channels*times
        fs: 采样频率
        lcf: 高通滤波频率
        hcf: 低通滤波频率
        order: 阶数, 默认为6阶
    """
    data = data_raw.copy()
    channels, times = data.shape
    if hcf > fs / 2:
        print("hcf > fs / 2, setting hcf = fs / 2")
        hcf = fs / 2

    if lcf <= 0 or lcf > fs / 2 or lcf >= hcf:
        print("lcf <= 0 or > fs / 2 or >= hcf, setting lcf = 2")
        lcf = 2

    cf1 = lcf / (fs / 2)
    cf2 = hcf / (fs / 2)
    B, A = butter(order, [cf1, cf2], "bandpass")

    for channel in range(channels):
        data[channel, :] = filtfilt(B, A, data[channel, :])

    return data

def filter_data_notch(notch_freq, Q, data, fs=256):   
    """
    Introduction:
        陷波
    Args:
        data shape: sample * channel
        notch_freq: 陷波频率
        Q: 陷波质量
        fs: 原始数据采样率
        return: 陷波处理后的数据
    """
    w0 = notch_freq/(fs/2)
    b, a = signal.iirnotch(w0=w0, Q=Q)
    filter_data = []
    for chan_index in np.arange(data.shape[1]):
        data_f = signal.filtfilt(b, a, data[:,chan_index])
        filter_data.append(data_f)
    
    return np.asarray(filter_data).T

def min_max_trial(trial):
    """
    Introduction:
        min_max归一化数据
    Args:
        shape: _ * sample * channels
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    return np.asarray([min_max_scaler.fit_transform(x) for x in trial])

def z_score_trial(trial):
    """
    Introduction:
        z_score 归一化数据
    Args:
        shape: _ * sample * channels
    """
    return np.asarray([preprocessing.scale(x) for x in trial])

def re_data_slide(trial, label, win_len, overlap, is_filter, norm_method):
    """
    Introduction:
        使用滑动窗口增强数据
    Args:
        trial shape: sample * channel
        label: 0, 1, 2
        win_len: slide windows length
        overlap: slide windows overlap rate
        is_filter: filter eeg data
        norm_method: "min_max" or "z_sorce" normalization data
        return: new_trial, new_label
    """
    if is_filter:
        trial = filter_data(1, 75, trial)
        trial = filter_data_notch(60, 5, trial)

    if overlap == 0:
        win_num = trial.shape[0]//win_len
        chans = trial.shape[1]
        used_point = win_num*win_len
        new_trial = trial[:used_point,:].reshape(win_num, win_len, chans)
        if win_num == 0:    # 对于不满足的用均值补齐算求
            new_trial = np.pad(new_trial, ((0,win_len - len(trial),(0,0))), mode='mean').reshape(-1,win_len,chans)
    else:
        start_index = end_index = 0
        step_len = int(win_len*(1-overlap))
        new_trial = []
        while end_index < len(trial)-win_len:
            end_index = start_index + win_len
            new_trial.append(trial[start_index:end_index])
            start_index += step_len
        new_trial = np.asarray(new_trial)

    if norm_method == "min_max":
        new_trial = min_max_trial(new_trial)

    if norm_method == "z_score":
        new_trial = z_score_trial(new_trial)
    
    new_label = np.asarray([label] * len(new_trial))
    
    return new_trial, new_label

def data_align(multimodel_list, fs_list):
    """
    Introduction:
        对齐一个trial的脑电和眼动数据
    Args:
        multimodel_list shape: multimodel list, shape: sample * channel
        fs_list: 不同模态采样率
        return: align_multimodel_list
    """
    assert len(multimodel_list) == len(fs_list), "multimodel list length not equal fs list"

    min_time = int(min([(len(multimodel_list[i]) / fs_list[i]) for i in range(len(multimodel_list))]))

    align_multimodel_list = []
    for i, model in enumerate(multimodel_list):
        align_multimodel_list.append(model[:min_time * fs_list[i]])

    return align_multimodel_list

def split_train_test_unimodal(data, label, mode, split_rate = 0.7, random_seed = 11):
    """
    Introduction:
        划分全部数据集为训练集和测试集
    Args:
        data shape: subject * trial * sample * channel
        label shape: subject * trail
        mode: "independent" or "dependent" split
        split_rate: 随机将原始数据集按split_rate分割为训练集和测试集
        random_seed: 随机种子
        return: train_data, train_label, test_data, test_label
    """
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if mode == "dependent":
        indices = list(range(len(data)))
        ## set random seed
        random.seed(random_seed) 
        random.shuffle(indices)
        split_idx = int(np.floor((1 - split_rate) * len(indices)))
        train_idx, test_idx = indices[split_idx:], indices[:split_idx]
        train_data, train_label = data[train_idx], label[train_idx]
        test_data, test_label = data[test_idx], label[test_idx]
        
    elif mode == "independent":
        for item in range(len(data)):
            indices = list(range(len(data[item])))
            random.seed(random_seed) 
            random.shuffle(indices)
            split_idx = int(np.floor((1 - split_rate) * len(indices)))
            train_idx, test_idx = indices[split_idx:], indices[:split_idx]
            train_data.append(data[item][train_idx])
            train_label.append(label[item][train_idx])
            test_data.append(data[item][test_idx])
            test_label.append(label[item][test_idx])

    train_data = np.concatenate(train_data, axis = 0)
    train_label = np.concatenate(train_label, axis = 0)
    test_data = np.concatenate(test_data, axis = 0)
    test_label = np.concatenate(test_label, axis = 0)

    return train_data, train_label, test_data, test_label