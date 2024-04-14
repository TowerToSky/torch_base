# TorchBase

使用Git上Pytorch通用训练框架来构建

This is a project about how to do nn experiments methodically with pytorch.

More instruction: [如何让炼丹更有条理](https://github.com/ahangchen/windy-afternoon/blob/master/ml/pratice/torch_best_practice.md)


# BaseModal

参考于上海交大吕宝梁组的EEG-Eye Movements Cross-Modal Decision Confidence Measurement with Generative Adversarial Networks一文，尝试去复现实现base结果，后续进行改进。

该proj分两阶段进行，一阶段主要以自编码器构成分别学习EEG和Eye的高层表示，采用拼接的方式实现多模态融合，然后进行分类任务，其loss为：

$$
\mathcal{L}_{RC} = \mathcal{L}_{MSE}(X_{eye},\hat{X}_{eye}) + \mathcal{L}_{MSE}(X_{eeg},\hat{X}_{eeg})\\
\\
\hat{y} = CLS( \hat{X}_{eye}, \hat{X}_{eeg} )\\
\\
\mathcal{L} = \lambda_{RC} \mathcal{L}_{RC} + \lambda_{CLS} \mathcal{L}_{CLS}
$$
其中$\mathcal{L}_{MSE}$为均方误差，$\mathcal{L}_{CLS}$为交叉熵损失，$\lambda_{RC}$和$\lambda_{CLS}$为权重。X为原始数据，$\hat{X}$为重构数据，CLS为分类器。

二阶段主要为将一阶段训练好的自编码器的编码器结构用于生成EEG和Eye的高层表示，然后将Eye的高层表示输入到生成器中，生成EEG的高层表示，然后分别将原始EEG高层表示和原始Eye高层表示融合，以及生成的EEG高层表示和原始Eye高层表示融合，最后送入到判别器中使两者表示尽可能相似。其loss为：



$$

\hat{O}_{eeg} = G(O_{eye}; \theta)\\

\mathcal{L}_D = \mathcal{L}_{CE}(D((O_{eye}, O_{eeg}); \beta), 1) \\

+ \mathcal{L}_{CE}(D((O_{eye}, \hat{O}_{eeg}); \beta), 0) \\

$$

判别器是一个二元分类器，$\beta$为判别器的参数，$\mathcal{L}_{CE}$为交叉熵损失。$G$为生成器，$\theta$为生成器的参数。$D$为判别器，$\mathcal{L}_D$为判别器的损失。$O_{eye}$为原始Eye高层表示，$O_{eeg}$为原始EEG高层表示，$\hat{O}_{eeg}$为生成的EEG高层表示。

通过固定判别器中参数$\beta$来训练以下目标函数$\mathcal{L}_G$的生成器：

$$

\mathcal{L}_G = \mathcal{L}_{CE}(D((O_{eye}, \hat{O}_{eeg}); \beta), 1) \\

$$

此外用内容损失函数保证生成的EEG高层表示与原始EEG高层表示尽可能相似，通过最小化它们之间欧几里得距离来实现，具体为：

$$

\mathcal{L}_{MES}(O_{eeg},\hat{O}_{eeg}) = ||O_{eeg} - G(O_{eye};\theta)||_2^2\\
$$

此时G的整体损失函数为：

$$

\mathcal{L} = \lambda_{MSE}\mathcal{L}_{MSE} + \lambda_{G}\mathcal{L}_G\\
$$

# 实验

## 论文

### 数据

论文中使用的为SEED-VPDC数据集，包含脑电合眼动，用于测量五级决策置信度。该实验有135次实验，每次实验一张图像，对应一个决策，共14名受试者，13名数据可用。

眼动信号，使用Tobii Pro提取22个特征，包含瞳孔直径、注视持续时间、眨眼持续时间和扫视持续时间。EEG信号由62导有源AgCl电极帽合ESI NeuroScan系统以10-20系统以1000Hz采集。每个通道用0.3-50Hz带通滤波，使用线性动态系统（LDS）平滑。在不重叠1s窗口从5个频带（δ：1-3Hz、θ：4-7Hz、α：8-13Hz、β：14-30Hz和γ：31-50Hz）提取DE（Different entropy，微分熵）特征。

### 实验设置

五折交叉验证方法，受试者依赖设置。并使用SVM合带有快捷连接的深度神经网络（DNNS）为基线模型。使用径向基函数核从$2^{-5:10}$的参数空间搜索SVM中的C。DNNS为4个隐藏层和1个输出层，隐藏层大小在16-256间搜索，lr为0.001。分别对EEG和Eye进行训练和测试。选择基于聚合的融合作为多模态融合策略，从DAE中提取高级脑电和眼动特征拼接。最后在SEED-VPDC数据集上测试【9】提出的基于深度对抗学习（DAL）的模型，该模型直接从主要眼动特征生成脑电特征。


# 本文设置

## 数据准备

采用空间能力测试数据集，其参考`SpatialAbilityTest/newEx1/`的曾经写过的来编写

采用1-36号数据，不包含9，11，15，19，20，33

### 脑电数据

脑电数据预处理方式为：0.5-70Hz滤波，50Hz陷波，重采样到256Hz，ICA去除Fp1，Fp2眼电的影响（TODO：可能没做），然后获取两个fixtion之间的数据作为受试者做题的数据，`滑窗1s，no-overlap`，计算`1-4,4-8,8-13,13-31,31-70`共5个频带上的`DE`特征，对每一个频带每一个通道上进行标准化和归一化，然后进行0填充到30长度（TODO：待研究为啥这样做，可能是为了补齐），然后保存成文件脑电特征提取代码位于：`/home/yihaoyuan/WorkSpace/SpatialAbilityTest/ExtractFeatures.ipynb`（有点乱，需要细心看）

脑电特征保存到`/data/SpaticalTest/ExtractedFeatures2/DE`，预先提取的5个频带的微分熵特征，其形状为`(29, 54, 31, 150)`，其中29为人数，54为题目数量，31为通道数，150为特征数量,

### 眼动数据

眼动数据的预处理文件位于：`/home/yihaoyuan/WorkSpace/SpatialAbilityTest/EyeTrack`，处理步骤为获取两个fixation之间眼动数据，移除Na，保存成每一题的眼动数据，对于每道题目数据去除类型为未识别的数据，对于Validity为空的进行插值，然后保留瞳孔直径、注视点、眼动类型和注视事件时间数据，然后以`win_len=60,overlap=0.25`滑窗截取，对于每一个滑窗提取眼动特征包含共`41`个特征。

```
blink_count, avg_blink_duration, std_blink_duration, min_blink_duration, max_blink_duration,

fixation_count, avg_fixation_duration, std_fixation_duration, min_fixation_duration, max_fixation_duration, avg_x, std_x, skew_x, kurt_x, avg_y, std_y, skew_y, kurt_y, avg_disper, std_disper, max_disper,

saccade_count, avg_saccade_duration, std_saccade_duration, min_saccade_duration, max_saccade_duration, avg_saccade_amplitude, std_saccade_amplitude, avg_saccade_velocity,

avg_lpupil, std_lpupil, min_lpupil, max_lpupil, skew_lpupil, kurt_lpupil, avg_rpupil, std_rpupil, min_rpupil, max_rpupil, skew_rpupil, kurt_rpupil
```


保存到`/data/SpaticalTest/eye_track_feature/`，为手动提取的41个特征，具体特征需看提取眼动特征的代码文件，形状为`（29，54，41）`，其中29为人数，54为题目数量，41为特征数量，眼动特征提取代码位于：`/home/yihaoyuan/WorkSpace/SpatialAbilityTest/EyeTrack/getEyeTrackDataFeatures.py`

### Label

标签来自`/home/yihaoyuan/WorkSpace/SpatialAbilityTest/ExtractFeatures.ipynb`，保存于`/data/SpaticalTest/ExtractedFeatures2/Labels/`，形状为`（29，54）`，29为人数，54为题目数量，label标签为：`0：困惑，1：猜测，2：非困惑，3：think-right`，（通常2分类需要的困惑和非困惑，但是这里是4分类，所以需要将think-right和非困惑合并为一类（提示给出的，可以考虑））

### 数据集划分

受试者依赖（被试内）：

每个人的数据先打乱，再7：3分，然后7放一块儿做训练集，3放一块儿做测试集

即（29，54，31，150）可划分为：

1、训练集：29，38，31\*150

2、测试集：29，16，31\*150

这里把最后两维合并成一维为31\*150表示特征，通道数即C均设置成1，表现为NCHW形状

然后进行5折交叉验证（也可以留一）