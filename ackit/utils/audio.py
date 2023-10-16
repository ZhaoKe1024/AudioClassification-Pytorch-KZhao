#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 12:23
# @Author: ZhaoKe
# @File : audio.py
# @Software: PyCharm
import io
import itertools
import os
import random
import wave

import av
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, samples, sr):
        self._samples = _convert_samples_to_float32(samples)
        self._sr = sr
        if self._samples.ndim >= 2:
            self._sample = np.mean(self._samples, axis=1)

    @classmethod
    def from_file(cls, file):
        assert os.path.exists(file), f"this file does not exists, please check the path: {file}"
        try:
            samples, sr = soundfile.read(file, dtype="float32")
        except:
            sr = 16000
            samples = decode_audio(file=file, sample_rate=sr)
        return cls(samples, sr)

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

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
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


def demo_plot_spec():
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold1/7061-6-0-0.wav"
    wav_data = get_spectrogram(file_path)
    plt.figure(0)
    # print(wav_data.shape)
    r, c = wav_data.shape
    plt.plot(range(c), wav_data[0,:], c='r')
    plt.plot(range(c), wav_data[1,:], c='b')
    plt.show()


if __name__ == '__main__':
    demo_plot_spec()