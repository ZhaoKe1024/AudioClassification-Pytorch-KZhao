# 声音分类


MFCC特征+TDNN
### 数据预处理过程：
1. Dataset 返回 (waveform, label), 其中waveform均通过mean操作合并为单通道,并且采样率均通过resample操作改为16000(因为UrbanSound8K数据集里面音频的长度,通道数,采样率都不一样,需要转换为一样的)
2. DataLoader 所使用的 collate_fn 为: 给定一个batch的waveform,求其中最长的长度,然后对其他waveform,在两端对称地填充0,并返回(waveforms, labels, len_ratios),此时DataLoader返回值是batch_size个(waveforms, labels, len_ratios).
3. 迭代DataLoader,将waveform通过Featurizer转换为MFCC特征,转换之后要根据collate_fn填充的长度进行掩码操作,填充的部分为False,返回0,原feature的部分为True,返回feature.

由于IO太慢跟不上GPU处理的速度,cuda利用率极低,考虑把数据预先处理为mfcc并保存文件,再用Dataset和DataLoader读取.

见create_data.py: 
- def create_file_list(metafile_path): 该函数创建train,valid,test数据集列表,比例8:1:1
- def create_mfcc_npy_data(root, tra_val="train"):该函数把上述数据转换为mfcc,根据最大长度对齐(两端对称0填充),然后存储为npy文件,把label和填充索引位置存为另一个npy文件,不再分批次读取音频并转换为MFCC并填充.运行此函数,在/datasets/下面可以找到生成的六个npy文件,分别是train,valid,test数据集的mfcc矩阵,label和填充起始位置列表.

### 训练

运行train_hst.py即可训练,训练中在”./runs/tdnn-MFCC/“文件夹下生成每个epoch中训练、验证的损失函数值、accuracy值，通过ploter.py读取并绘制折线图。

为了提高数据的质量,之后打算裁剪掉静默的音频段,然后再做MFCC. 在utils/audio.py中，AudioSegment类实现了vad()和add_noise()函数。

流程已实现，但是由于以上预处理要把所有的音频转换为MFCC然后存为一个npy文件，由于内存无法装下一次性处理过的这么多文件，因此处理过程中最好边写入边处理，方法是，把每一条音频的MFCC设置为固定shape(40, 174)，展开为一维向量，和标签一起写入txt文件中，读取的时候再reshape(40, 174)即可。txt文件中每行2+40*174列，分别是 label, pad_start, mfcc。

统计发现，噪声文件中，各个声道数的音频数目为：{1: 20881, 2: 107, 8: 126, 16: 75, 30: 71}。因此在添加噪声的过程中可能会有内存爆满的情况，因此读取噪声文件后应将其取其中一个声道。


# Reference
1. 参考其代码结构: https://github.com/yeyupiaoling/AudioClassification-Pytorch