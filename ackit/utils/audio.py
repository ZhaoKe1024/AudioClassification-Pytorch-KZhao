#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 12:23
# @Author: ZhaoKe
# @File : audio.py
# @Software: PyCharm
import copy
import io
import itertools
import os
import random
import wave

import av
import librosa
import matplotlib.pyplot as plt
import numpy as np
import resampy
import soundfile


def _convert_samples_to_float32(samples):
    """Convert sample type to float32.

    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= (1. / 2 ** (bits - 1))
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples


class AudioSegment(object):
    def __init__(self, samples, sample_rate, resample=False, res_sr=16000):
        self._samples = _convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._sample = np.mean(self._samples, axis=1)
        if resample:
            self.resample(target_sample_rate=res_sr)

    @property
    def samples(self):
        return self._samples

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @classmethod
    def from_file(cls, file, resample=False, res_sr=16000):
        assert os.path.exists(file), f"this file does not exists, please check the path: {file}"
        try:
            samples, sr = soundfile.read(file, dtype="float32")
        except:
            sr = 16000
            samples = decode_audio(file=file, sample_rate=sr)
        return cls(samples, sr, resample=resample, res_sr=res_sr)

    def normalize(self, target_db=-20, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)

        :param target_db: Target RMS value in decibels. This value should be
                          less than 0.0 as 0.0 is full-scale audio.
        :type target_db: float
        :param max_gain_db: Max amount of gain in dB that can be applied for
                            normalization. This is to prevent nans when
                            attempting to normalize a signal consisting of
                            all zeros.
        :type max_gain_db: float
        :raises ValueError: If the required gain to normalize the segment to
                            the target_db value exceeds max_gain_db.
        """
        if -np.inf == self.rms_db: return
        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError(
                "无法将段规范化到 %f dB，因为可能的增益已经超过max_gain_db (%f dB)" % (target_db, max_gain_db))
        self.gain_db(min(max_gain_db, target_db - self.rms_db))

    @property
    def rms_db(self):
        """返回以分贝为单位的音频均方根能量

        :return: Root mean square energy in decibels.
        :rtype: float
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    def gain_db(self, gain):
        """对音频施加分贝增益。

        Note that this is an in-place transformation.

        :param gain: Gain in decibels to apply to samples.
        :type gain: float|1darray
        """
        self._samples *= 10. ** (gain / 20.)

    def resample(self, target_sample_rate, filter='kaiser_best'):
        """按目标采样率重新采样音频

        Note that this is an in-place transformation.
        :param target_sample_rate: Target sample rate.
        :type target_sample_rate: int
        :param filter: The resampling filter to use one of {'kaiser_best', 'kaiser_fast'}.
        :type filter: str
        """
        # print(self.samples.shape)
        if self._samples.ndim > 2 or self._samples.ndim < 1:
            raise ValueError("the dimension of sound is more than 2.")
        if self._samples.ndim == 2:
            new_sample = []
            for i in range(self._samples.shape[1]):
                new_sample.append(
                    resampy.resample(self._samples[:, i], self._sample_rate, target_sample_rate, filter=filter))
            self._samples = np.array(new_sample)
        else:
            self._samples = resampy.resample(self._samples, self._sample_rate, target_sample_rate, filter=filter)
        self._sample_rate = target_sample_rate

    def vad(self, top_db=20, overlap=200):
        self._sample = vad(self._sample, top_db=top_db, overlap=overlap)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        """只加载一小段音频，而不需要将整个文件加载到内存中，这是非常浪费的。

        :param file: 输入音频文件路径或文件对象
        :type file: str|file
        :param start: 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
        :type start: float
        :param end: 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。
        :type end: float
        :return: AudioSegment输入音频文件的指定片的实例。
        :rtype: AudioSegment
        :raise ValueError: 如开始或结束的设定不正确，例如时间不允许。
        """
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = 0. if start is None else round(start, 3)
        end = duration if end is None else round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return cls(data, sample_rate)

    def add_noise(self,
                  noise,
                  snr_dB,
                  max_gain_db=300.0):
        """以特定的信噪比添加给定的噪声段。如果噪声段比该噪声段长，则从该噪声段中采样匹配长度的随机子段。

        Note that this is an in-place transformation.

        :param noise: Noise signal to add.
        :type noise: AudioSegment
        :param snr_dB: Signal-to-Noise Ratio, in decibels.
        :type snr_dB: float
        :param max_gain_db: Maximum amount of gain to apply to noise signal
                            before adding it in. This is to prevent attempting
                            to apply infinite gain to a zero signal.
        :type max_gain_db: float
        :raises ValueError: If the sample rate does not match between the two
                            audio segments, or if the duration of noise segments
                            is shorter than original audio segments.
        """
        if noise._sample_rate != self._sample_rate:
            raise ValueError("噪声采样率(%d Hz)不等于基信号采样率(%d Hz)" % (noise._sample_rate, self._sample_rate))
        if noise.duration < self.duration:
            raise ValueError("噪声信号(%f秒)必须至少与基信号(%f秒)一样长" % (noise.duration, self.duration))
        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)
        noise_new = copy.deepcopy(noise)
        noise_new.random_subsegment(self.duration)
        noise_new.gain_db(noise_gain_db)
        # print("self sample: ", self.samples.shape)
        # print("other sample:", noise_new.samples.shape)
        self.superimpose(noise_new)

    def random_subsegment(self, subsegment_length):
        """随机剪切指定长度的音频片段

        Note that this is an in-place transformation.

        :param subsegment_length: Subsegment length in seconds.
        :type subsegment_length: float
        :raises ValueError: If the length of subsegment is greater than
                            the origineal segemnt.
        """
        if subsegment_length > self.duration:
            raise ValueError("Length of subsegment must not be greater "
                             "than original segment.")
        start_time = random.uniform(0.0, self.duration - subsegment_length)
        self.subsegment(start_time, start_time + subsegment_length)

    def subsegment(self, start_sec=None, end_sec=None):
        """在给定的边界之间切割音频片段

        Note that this is an in-place transformation.

        :param start_sec: Beginning of subsegment in seconds.
        :type start_sec: float
        :param end_sec: End of subsegment in seconds.
        :type end_sec: float
        :raise ValueError: If start_sec or end_sec is incorrectly set, e.g. out
                           of bounds in time.
        """
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        if start_sec < 0.0:
            raise ValueError("切片起始位置(%f s)越界" % start_sec)
        if end_sec < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end_sec)
        if start_sec > end_sec:
            raise ValueError("切片的起始位置(%f s)晚于结束位置(%f s)" % (start_sec, end_sec))
        if end_sec > self.duration:
            raise ValueError("切片结束位置(%f s)越界(> %f s)" % (end_sec, self.duration))
        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

    def superimpose(self, other):
        """将另一个段的样本添加到这个段的样本中(以样本方式添加，而不是段连接)。

        :param other: 包含样品的片段被添加进去
        :type other: AudioSegments
        :raise TypeError: 如果两个片段的类型不匹配
        :raise ValueError: 不能添加不同类型的段
        """
        if not isinstance(other, type(self)):
            raise TypeError("不能添加不同类型的段: %s 和 %s" % (type(self), type(other)))
        if self._sample_rate != other._sample_rate:
            raise ValueError("采样率必须匹配才能添加片段")
        if len(self._samples) != len(other._samples):
            raise ValueError("段长度必须匹配才能添加段")
        self._samples += other._samples


def decode_audio(file, sample_rate: int = 16000):
    """读取音频，主要用于兜底读取，支持各种数据格式

    Args:
      file: Path to the input file or a file-like object.
      sample_rate: Resample the audio to this sample rate.

    Returns:
      A float32 Numpy array.
    """
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(file, metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    return audio.astype(np.float32) / 32768.0


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo._samples >= num_samples:
            yield fifo.read()

    if fifo._samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def get_spectrogram(file_path):
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    n_channel, sample_width, frame_rate, n_frames = params[:4]
    str_data = f.readframes(n_frames)
    wav_data = np.fromstring(str_data, dtype=np.short)
    wav_data = wav_data * 1.0 / max(abs(wav_data))
    wav_data = np.reshape(wav_data, [n_frames, n_channel]).T
    f.close()

    # print(get_spectrogram())
    frame_length = 0.025
    frame_size = frame_length * frame_rate

    return wav_data


def vad(wav, top_db=20, overlap=200):
    # print("wav.shape: ", wav.shape)
    # Split an audio signal into non-silent intervals
    intervals = librosa.effects.split(wav, top_db=top_db)
    # print("intervals:\n", intervals)
    if len(intervals) == 0:
        return wav
    # wav_output = [np.array([])]
    wav_output = []
    for sliced in intervals:
        seg = wav[sliced[0]:sliced[1]]
        # print("seg:\n", seg)
        if len(wav_output) == 0:
            wav_output = [seg]
        if len(seg) < 2 * overlap:
            wav_output[-1] = np.concatenate((wav_output[-1], seg))
        else:
            wav_output.append(seg)
    wav_output = [x for x in wav_output if len(x) > 0]

    if len(wav_output) == 1:
        wav_output = wav_output[0]
    else:
        wav_output = concatenate(wav_output)
    return wav_output


def concatenate(wave, overlap=200):
    total_len = sum([len(x) for x in wave])
    unfolded = np.zeros(total_len)

    # Equal power crossfade
    window = np.hanning(2 * overlap)
    fade_in = window[:overlap]
    fade_out = window[-overlap:]

    end = total_len
    for i in range(1, len(wave)):
        prev = wave[i - 1]
        curr = wave[i]

        if i == 1:
            end = len(prev)
            unfolded[:end] += prev

        max_idx = 0
        max_corr = 0
        pattern = prev[-overlap:]
        # slide the curr batch to match with the pattern of previous one
        for j in range(overlap):
            match = curr[j:j + overlap]
            corr = np.sum(pattern * match) / [(np.sqrt(np.sum(pattern ** 2)) * np.sqrt(np.sum(match ** 2))) + 1e-8]
            if corr > max_corr:
                max_idx = j
                max_corr = corr

        # Apply the gain to the overlap samples
        start = end - overlap
        unfolded[start:end] *= fade_out
        end = start + (len(curr) - max_idx)
        curr[max_idx:max_idx + overlap] *= fade_in
        unfolded[start:end] += curr[max_idx:]
    return unfolded[:end]


def augment_audio(noises_path,
                  audio_segment,
                  speed_perturb=False,
                  volume_perturb=False,
                  volume_aug_prob=0.2,
                  noise_dir=None,
                  noise_aug_prob=0.2):
    # 语速增强，注意使用语速增强分类数量会大三倍
    if speed_perturb:
        speeds = [1.0, 0.9, 1.1]
        speed_idx = random.randint(0, 2)
        speed_rate = speeds[speed_idx]
        if speed_rate != 1.0:
            audio_segment.change_speed(speed_rate)
    # 音量增强
    if volume_perturb and random.random() < volume_aug_prob:
        min_gain_dBFS, max_gain_dBFS = -15, 15
        gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
        audio_segment.gain_db(gain)
    # 获取噪声文件
    if noises_path is None and noise_dir is not None:
        noises_path = []
        if noise_dir is not None and os.path.exists(noise_dir):
            for file in os.listdir(noise_dir):
                noises_path.append(os.path.join(noise_dir, file))
    # 噪声增强
    if len(noises_path) > 0 and random.random() < noise_aug_prob:
        min_snr_dB, max_snr_dB = 10, 50
        # 随机选择一个noises_path中的一个
        noise_path = random.sample(noises_path, 1)[0]
        # print(noise_path)
        # 读取噪声音频
        noise_segment = AudioSegment.slice_from_file(noise_path)
        # 如果噪声采样率不等于audio_segment的采样率，则重采样
        if noise_segment._sample_rate != audio_segment._sample_rate:
            noise_segment.resample(audio_segment._sample_rate)
        # 随机生成snr_dB的值
        snr_dB = random.uniform(min_snr_dB, max_snr_dB)
        # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
        if noise_segment.duration < audio_segment.duration:
            diff_duration = audio_segment.num_samples - noise_segment.num_samples
            noise_segment._samples = np.pad(noise_segment._samples, (0, diff_duration), 'wrap')
        # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
        audio_segment.add_noise(noise_segment, snr_dB)
    return audio_segment


def demo_plot_spec():
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold1/7061-6-0-0.wav"
    wav_data = get_spectrogram(file_path)
    plt.figure(0)
    # print(wav_data.shape)
    r, c = wav_data.shape
    plt.plot(range(c), wav_data[0, :], c='r')
    plt.plot(range(c), wav_data[1, :], c='b')
    plt.show()


if __name__ == '__main__':
    demo_plot_spec()
