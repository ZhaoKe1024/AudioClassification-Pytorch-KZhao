# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-26 0:53
import random
import numpy as np
import librosa


class AudioSegment(object):
    """Monaural audio segment abstraction.

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate):
        """Create audio segment from samples.

        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        self._samples = _convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def vad(self, top_db=20, overlap=200):
        self._samples = vad(wav=self._samples, top_db=top_db, overlap=overlap)

    def gain_db(self, gain):
        """对音频施加分贝增益。

        Note that this is an in-place transformation.

        :param gain: Gain in decibels to apply to samples.
        :type gain: float|1darray
        """
        self._samples *= 10. ** (gain / 20.)

    def change_speed(self, speed_rate):
        """通过线性插值改变音频速度

        :param speed_rate: Rate of speed change:
                           speed_rate > 1.0, speed up the audio;
                           speed_rate = 1.0, unchanged;
                           speed_rate < 1.0, slow down the audio;
                           speed_rate <= 0.0, not allowed, raise ValueError.
        :type speed_rate: float
        :raises ValueError: If speed_rate <= 0.0.
        """
        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = self._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self._samples = np.interp(new_indices, old_indices, self._samples).astype(np.float32)

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
        if -np.inf == self.rms_db:
            return
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


def wav_slice_padding(old_signal, save_len=160000):
    new_signal = np.zeros(save_len)
    if old_signal.shape[0] < save_len:
        resi = save_len - old_signal.shape[0]
        # print("resi:", resi)
        new_signal[:old_signal.shape[0]] = old_signal
        new_signal[old_signal.shape[0]:] = old_signal[-resi:][::-1]
    elif old_signal.shape[0] > save_len:
        posi = random.randint(0, old_signal.shape[0] - save_len)
        new_signal = old_signal[posi:posi+save_len]
    return new_signal


def vad(wav, top_db=20, overlap=200):
    # Split an audio signal into non-silent intervals
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    wav_output = [np.array([])]
    for sliced in intervals:
        seg = wav[sliced[0]:sliced[1]]
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
