import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import collections
from sklearn import preprocessing


## 时域特征

def find_type(self,type):

    onset = []
    offset = []

    i = 0
#     n = len(self) - 1
    
    while i < len(self):
        if self['Eye_movement_type'][i] == type:
            onset.append(i)
            while i < len(self) and self['Eye_movement_type'][i] == type:
                i += 1
            offset.append(i-1)
        else:
            i += 1

    indices = {"start": onset, "end": offset}

    return indices

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

def slide_trial(data, wid_len, overlap, is_slide):
    
    new_data = []
    
    if is_slide:    
        step_len = int(wid_len*(1-overlap))
        win_len = (len(data) - wid_len) // step_len
        for wid_index in np.arange((len(data)-wid_len)//step_len):
            new_data.append(data[wid_index*step_len:wid_index*step_len+wid_len,:])
        
        # 为最后一个窗口补0
        if len(data)-wid_len - win_len*step_len > 0:
            new_data.append(np.concatenate((data[wid_len + win_len*step_len:,:],np.zeros((2*wid_len - len(data) + win_len*step_len, data.shape[1])))))
    else:
        new_data.append(data)  
        
    return np.array(new_data),np.array(new_data).shape

# def findBlinkParams(self):
    
#     indices = find_type(self,0)
#     onset = indices["start"]
#     offset = indices["end"]
    
#     count = 0
#     duration = []
        
#     if 0 in self.Eye_movement_type.values:

#         for start, end in zip(onset, offset):

#             dur = self.loc[start, "Gaze_event_duration"]
#             duration.append(dur)

#             count += 1

#         avg_dur = np.mean(duration)
#         std_dur = np.std(duration)
#         min_dur = np.min(duration)
#         max_dur = np.max(duration)
# #         skew_dur = pd.Series(duration).skew()
# #         kurt_dur = pd.Series(duration).kurt() 
    
#     else:
#         avg_dur, std_dur, min_dur, max_dur = 0, 0, 0, 0

#     return [count, avg_dur, std_dur, min_dur, max_dur]

def findFixationParams(self):
    
    indices = find_type(self,1)
    onset = indices["start"]
    offset = indices["end"]
    
    count = 0
    duration = []
    dispersion = []
    gaze_x = []
    gaze_y = []
    
    if 1 in self.Eye_movement_type.values:
        
        for start, end in zip(onset, offset):

            x_seq = self.loc[start:end, "Gaze_point_X"]
            y_seq = self.loc[start:end, "Gaze_point_Y"]

            gaze_x.append(np.array(x_seq))
            gaze_y.append(np.array(y_seq))

    #             avg_x_seq = x_seq.mean(axis = 1)
    #             std_x_seq = x_seq.std(axis = 1)
    #             skew_x_seq = x_seq.skew(axis = 1)
    #             kurt_x_seq = x_seq.kurt(axis = 1)
    #             avg_y_seq = y_seq.mean(axis = 1)
    #             std_y_seq = y_seq.std(axis = 1)
    #             skew_y_seq = y_seq.skew(axis = 1)
    #             kurt_y_seq = y_seq.kurt(axis = 1)

            disper = (np.max(x_seq) -  np.min(x_seq)) + (np.max(y_seq) -  np.min(y_seq))
            dispersion.append(disper)

            dur = self.loc[start, "Gaze_event_duration"]
            duration.append(dur)

            count += 1

        avg_x = np.mean(np.concatenate(gaze_x))
        std_x = np.std(np.concatenate(gaze_x))
        min_x = np.min(np.concatenate(gaze_x))
        max_x = np.max(np.concatenate(gaze_x))
        skew_x = pd.Series(np.concatenate(gaze_x)).skew()
        kurt_x = pd.Series(np.concatenate(gaze_x)).kurt() 

        avg_y = np.mean(np.concatenate(gaze_y))
        std_y = np.std(np.concatenate(gaze_y))
        min_y = np.min(np.concatenate(gaze_y))
        max_y = np.max(np.concatenate(gaze_y))
        skew_y = pd.Series(np.concatenate(gaze_y)).skew()
        kurt_y = pd.Series(np.concatenate(gaze_y)).kurt()

        avg_dur = np.mean(duration)
        std_dur = np.std(duration)
        min_dur = np.min(duration)
        max_dur = np.max(duration)
    #         skew_dur = pd.Series(duration).skew()
    #         kurt_dur = pd.Series(duration).kurt()       

        avg_disper = np.mean(dispersion)
        std_disper = np.std(dispersion)
        max_disper = np.max(dispersion)
        
    else:
        count, avg_dur, std_dur, min_dur, max_dur, avg_x, std_x, skew_x, kurt_x, avg_y, std_y, skew_y, kurt_y, avg_disper, std_disper, max_disper = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
        
    return [count, avg_dur, std_dur, min_dur, max_dur, avg_x, std_x, skew_x, kurt_x, avg_y, std_y, skew_y, kurt_y, avg_disper, std_disper, max_disper]

def findSaccadeParams(self):

    indices = find_type(self,2)
    onset = indices["start"]
    offset = indices["end"]

    count = 0
    duration = []
    amplitude = []
    velocity = []
    
    n = len(self) - 1

    if 2 in self.Eye_movement_type.values:
        
        for start, end in zip(onset, offset):
            
            if start == 0:
                pre_x, pre_y = self.loc[start, "Gaze_point_X"], self.loc[start, "Gaze_point_Y"] 
                aft_x, aft_y = self.loc[end + 1 , "Gaze_point_X"], self.loc[end + 1 , "Gaze_point_Y"]
                dis = np.sqrt((aft_x - pre_x) ** 2 + (aft_y - pre_y) ** 2)
                amplitude.append(dis)

                dur = self.loc[start, "Gaze_event_duration"]
                duration.append(dur)

                vel = dis / dur
                velocity.append(vel)

                count += 1
                
            elif end == n:
                pre_x, pre_y = self.loc[start - 1, "Gaze_point_X"], self.loc[start - 1, "Gaze_point_Y"] 
                aft_x, aft_y = self.loc[end, "Gaze_point_X"], self.loc[end, "Gaze_point_Y"]
                dis = np.sqrt((aft_x - pre_x) ** 2 + (aft_y - pre_y) ** 2)
                amplitude.append(dis)

                dur = self.loc[start, "Gaze_event_duration"]
                duration.append(dur)

                vel = dis / dur
                velocity.append(vel)

                count += 1

            else:

                pre_x, pre_y = self.loc[start - 1 , "Gaze_point_X"], self.loc[start - 1 , "Gaze_point_Y"] 
                aft_x, aft_y = self.loc[end + 1 , "Gaze_point_X"], self.loc[end + 1 , "Gaze_point_Y"]
                dis = np.sqrt((aft_x - pre_x) ** 2 + (aft_y - pre_y) ** 2)
                amplitude.append(dis)

                dur = self.loc[start, "Gaze_event_duration"]
                duration.append(dur)

                vel = dis / dur
                velocity.append(vel)

                count += 1

        avg_dur = np.mean(duration)
        std_dur = np.std(duration)
        min_dur = np.min(duration)
        max_dur = np.max(duration)
    #     skew_dur = pd.Series(duration).skew()
    #     kurt_dur = pd.Series(duration).kurt()

        avg_ampl = np.mean(amplitude)
        std_ampl = np.std(amplitude)

        avg_vel = np.mean(velocity)
        
    else:
        count, avg_dur, std_dur, min_dur, max_dur, avg_ampl, std_ampl, avg_vel = 0, 0, 0, 0, 0, 0, 0, 0

    return [count, avg_dur, std_dur, min_dur, max_dur, avg_ampl, std_ampl, avg_vel]

def findPupilParams(self):
    
    avg_lpupil = self.loc[:,"Pupil_diameter_left"].mean()
    std_lpupil = self.loc[:,"Pupil_diameter_left"].std()
    min_lpupil = self.loc[:,"Pupil_diameter_left"].min()
    max_lpupil = self.loc[:,"Pupil_diameter_left"].max()   
    skew_lpupil = self.loc[:,"Pupil_diameter_left"].skew() # 求偏度
    kurt_lpupil = self.loc[:,"Pupil_diameter_left"].kurt() # 求峰度
    
    avg_rpupil = self.loc[:,"Pupil_diameter_right"].mean()
    std_rpupil = self.loc[:,"Pupil_diameter_right"].std()
    min_rpupil = self.loc[:,"Pupil_diameter_right"].min()
    max_rpupil = self.loc[:,"Pupil_diameter_right"].max()   
    skew_rpupil = self.loc[:,"Pupil_diameter_right"].skew()
    kurt_rpupil = self.loc[:,"Pupil_diameter_right"].kurt()
    
    return [avg_lpupil, std_lpupil, min_lpupil, max_lpupil, skew_lpupil, kurt_lpupil, avg_rpupil, std_rpupil, min_rpupil, max_rpupil, skew_rpupil, kurt_rpupil]    


def findBlinkParams(self):

    indices = find_type(self, 0)
    onset = indices["start"]
    offset = indices["end"]

    count = 0
    duration = []

    if 0 in self.Eye_movement_type.values:

        for start, end in zip(onset, offset):

            dur = self.loc[start, "Gaze_event_duration"]
            duration.append(dur)

            count += 1

        avg_dur = np.mean(duration)
        std_dur = np.std(duration)
        min_dur = np.min(duration)
        max_dur = np.max(duration)
#         skew_dur = pd.Series(duration).skew()
#         kurt_dur = pd.Series(duration).kurt()

    else:
        avg_dur, std_dur, min_dur, max_dur = 0, 0, 0, 0

    return [count, avg_dur, std_dur, min_dur, max_dur]


