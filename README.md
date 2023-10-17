# 声音分类


TDNN+MFCC特征（零填充collate）
数据预处理过程：
1. Dataset 返回 (waveform, label), 其中waveform均通过mean操作合并为单通道,并且采样率均通过resample操作改为16000(因为UrbanSound8K数据集里面音频的长度,通道数,采样率都不一样,需要转换为一样的)
2. DataLoader 所使用的 collate_fn 为: 给定一个batch的waveform,求其中最长的长度,然后对其他waveform,在两端对称地填充0,并返回(waveforms, labels, len_ratios),此时DataLoader返回值是batch_size个(waveforms, labels, len_ratios).
3. 迭代DataLoader,将waveform通过Featurizer转换为MFCC特征,转换之后要根据collate_fn填充的长度进行掩码操作,填充的部分为False,返回0,原feature的部分为True,返回feature.

由于IO太慢跟不上GPU处理的速度,cuda利用率极低,考虑一下把数据预先处理为mfcc并保存文件,再用Dataset和DataLoader读取.

见create_data.py: 
- def create_file_list(metafile_path): 该函数创建train,valid,test数据集列表,比例8:1:1
- def create_mfcc_npy_data(root, tra_val="train"):该函数把上述数据转换为mfcc,根据最大长度对齐(两端对称0填充),然后存储为npy文件,把label和填充索引位置存为另一个npy文件,不再分批次读取音频并转换为MFCC并填充.运行此函数,在/datasets/下面可以找到生成的六个npy文件,分别是train,valid,test数据集的mfcc矩阵,label和填充起始位置列表.

为了提高数据的质量,之后打算裁剪掉静默的音频段,然后再做MFCC,未完待续

# Reference
1. 参考其代码结构: https://github.com/yeyupiaoling/AudioClassification-Pytorch