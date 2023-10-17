# 声音分类（未完成）


TDNN+MFCC特征（零填充collate）
数据预处理过程：
1. Dataset 返回 (waveform, label), 其中waveform均通过mean操作合并为单通道,并且采样率均通过resample操作改为16000(因为UrbanSound8K数据集里面音频的长度,通道数,采样率都不一样,需要转换为一样的)
2. DataLoader 所使用的 collate_fn 为: 给定一个batch的waveform,求其中最长的长度,然后对其他waveform,在两端对称地填充0,并返回(waveforms, labels, len_ratios),此时DataLoader返回值是batch_size个(waveforms, labels, len_ratios).
3. 迭代DataLoader,将waveform通过Featurizer转换为MFCC特征,转换之后要根据collate_fn填充的长度进行掩码操作,填充的部分为False,返回0,原feature的部分为True,返回feature.

# Reference
1. 参考其代码结构: https://github.com/yeyupiaoling/AudioClassification-Pytorch