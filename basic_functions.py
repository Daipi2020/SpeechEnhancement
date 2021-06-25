"""
@FileName: basic_functions.py
@Description: Implement basic_functions
@Author  : Ryuk
@CreateDate: 2019/11/13 13:47
@LastEditTime: 2020/05/16 13:47
@LastEditors: Please set LastEditors
@Version: v1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy import signal
from numpy.linalg import norm
import librosa
from librosa import display
from scipy.io import wavfile
# from pesq import pesq

def sgn(data):
    """
    sign function
    :param data:
    :return: sign
    """
    if data >= 0 :
        return 1
    else:
        return 0

def normalization(data):
    """
    normalize data into [-1, 1]
    :param data: input data
    :return: normalized data
    """
    normalized_data = 2 * (data - min(data)) / (max(data) - min(data)) - 1
    return normalized_data

def preEmphasis(samples, fs, alpha=0.9375, overlapping=0, window_length=240, window_type='Rectangle', display=False):
    """
    per emphasis speech
    :param samples: sample data
    :param fs: sample frequency
    :param alpha: parameter
    :param overlapping: overlapping length
    :param window_length: the length of window
    :param window_type: the type of window
    :param display: whether to display processed speech
    :return: processed speech
    """
    y = np.zeros(len(samples))
    y[0] = samples[0]

    # pre emphasis
    for i in range(1, len(samples)):
        y[i] = samples[i] - alpha * samples[i-1]

    if display:
        time = np.arange(0, len(samples)) * (1.0 / fs)
        plt.plot(time, samples)
        plt.title("Pre-emphasis")
        plt.ylabel("Waveform")
        plt.xlabel("Time (seconds)")
        plt.show()

    return y


def displaySpeech(samples, fs):
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    time = np.arange(0, len(samples)) * (1.0 / fs)

    plt.plot(time, samples)
    plt.title("Speech")
    plt.xlabel("time (seconds)")
    plt.show()


def pesqTest(raw_wav_path, deg_wav_path, fs):
    """
    pesq test
    :param raw_wav_path: raw speech samples file path  原始语音样本
    :param deg_wav_path: degradation speech samples file path 退化的语音样本
    :param fs: sample frequency
    :return: save pesq value in current fold pesq_result.txt
    """
    pesq_exe = "./pesq.exe"
    commad = str('+') + str(fs) + ' ' + raw_wav_path + ' ' + deg_wav_path
    subprocess.Popen(pesq_exe + ' ' + commad)

def resample_rate(path, new_sample_rate):
    '''
    change sample_rate
    '''
    signal, sr = librosa.load(path, sr=None)

    # 提取wav文件的文件名
    #for example: wavfile path is  d:/audio/sample.wav
    wavfile = path.split('/')[-1]
    #then,  wavfile = sample.wav
    wavfile = wavfile.split('.')[0]
    #then,  wavfile = sample

    # 提取为重采样后的wav文件命名
    file_name = str(wavfile) + '_' + str(new_sample_rate) + '.wav'
    #then,  file_name = sample_new.wav
    new_signal = librosa.resample(signal, sr, new_sample_rate, res_type='kaiser_best')   #
    librosa.output.write_wav(file_name, new_signal, new_sample_rate)


def addNoise(origin_signal_path, noise_path, snr, display=False):
    """
    add noise with specific snr
    :param origin_signal_path: origin speech path
    :param noise_path: noise path
    :param snr: snr
    :param display: whether to display processed speech
    :return: mix speech
    """

    # 从路径中提取wav文件的文件名
    # for example: wavfile path is  d:/audio/dog.wav
    origin_whole_name = origin_signal_path.split('/')[-1]
    noise_whole_name = noise_path.split('/')[-1]
    # then,  origin_whole_name = dog.wav
    origin_name = origin_whole_name.split('.')[0]
    noise_name = noise_whole_name.split('.')[0]
    # then,  origin = dog

    #读取音频，
    origin, sr1 = librosa.load(origin_signal_path, sr=None)    # sr=None 以原始采样率读取音频
    #返回的origin是ndarray数组，sr1是int整型
    noise, sr2 = librosa.load(noise_path, sr=None)

    #判断采样率是否一致
    if sr1 != sr2:
        print('sr1 ≠ sr2' + ',' + 'addNoise error')
        print('original_signal_samplerate=' + str(sr1))
        print('noise_samplerate=' + str(sr2))
        print('The sampling rate should be the same')
    else:
        sr = sr1
        print('sr=' + str(sr))
        # 根据待加噪的信号与 噪声信号的长度，进行不同的处理
        if len(noise) > len(origin):
            len_diff = len(noise) - len(origin)
            l0 = random.randint(0, len_diff-1)
            noise = noise[l0:len(origin)]
        else:
            times = len(origin) // len(noise)  # 取整
            noise = np.tile(noise, times)  # 噪声序列重复次数 times
            # 噪声序列重复多次后与待加噪信号还存在长度差
            tail_len = len(origin) - len(noise)  # 噪声序列重复多次后，与原始信号还存在的长度差
            # 可以采取两种处理方法：补零 或者 补部分噪声序列，使得 clean 和 noise两个序列长度相等
            ## padding = [0] * (len(origin) - len(noise))  # 补 零序列
            padding = noise[0:tail_len]   # 补 部分噪声序列
            noise = np.hstack([noise, padding])  # 将补充的序列 拼接在noise序列后面

        noise = noise / norm(noise) * norm(origin) / (10.0 ** (0.05 * snr))
        mix = origin + noise
        output_filename = origin_name + '_' + noise_name + '_' + str(snr) + '.wav'
        librosa.output.write_wav(output_filename, mix, sr)

        if display:
            time = np.arange(0, len(origin)) * (1.0 / sr)

            plt.subplot(2, 1, 1)
            plt.plot(time, origin)
            plt.title("Original Speech")
            plt.xlabel("time (seconds)")

            plt.subplot(2, 1, 2)
            plt.plot(time, mix)
            plt.title("Mixed Speech")
            plt.xlabel("time (seconds)")
            plt.show()

        return mix


def addEcho(clean, sr, alpha, beta=0.5, delay=0.1, type=1):
    """
    add echo signal to raw speech
    :param clean: clean speech
    :param sr: sample rate
    :param alpha: parameters for type1
    :param beta: parameters for type2
    :param delay: parameters for type2
    :param type: echo type
    :return: mix signal
    """
    if type == 1:
        h = [1]
        h.extend([0] * int(alpha * sr))
        h.extend([0.5])
        h.extend([0] * int(alpha * sr))
        h.extend([0.25])
        mix = signal.convolve(clean, h)
    else:
        mix = clean.copy()
        shift = int(delay * sr)
        for i in range(shift, len(clean)):
            mix[i] = beta * clean[i] + (1 - beta) * clean[i - shift]
    return mix


def addReverberation(clean, alpha=0.8, R=2000):
    """
    add reverberation
    :param clean: clean speech
    :param alpha: factor
    :param R:
    :return: mix speech
    """
    b = [0] * (R+1)
    b[0], b[-1] = alpha, 1
    a = [0] * (R+1)
    a[0], a[-1] = 1, 0.5
    mix = signal.filtfilt(b, a, clean)
    return mix

def addHowl(clean, K=0.2):
    """
    add howl
    :param clean: clean speech
    :param K: factors
    :return: mix speech
    """
    g = np.loadtxt("../tool/path.txt")
    c = np.array([0, 0, 0, 0, 1]).T
    h = np.zeros(201)
    h[100] = 1

    xs1 = np.zeros(c.shape[0])
    xs2 = np.zeros(g.shape)
    xs3 = np.zeros(h.shape)

    mix = np.zeros(len(clean))
    temp = 0

    for i in range(len(clean)):
        xs1[1:] = xs1[: - 1]
        xs1[0] =  clean[i] + temp
        mix[i] = K * np.dot(xs1.T, c)

        xs3[1:] = xs3[: - 1]
        xs3[0] = mix[i]
        mix[i] = np.dot(xs3.T, h)

        mix[i] = min(1, mix[i])
        mix[i] = max(-1, mix[i])

        xs2[1:] = xs2[: - 1]
        xs2[0] = mix[i]
        temp = np.dot(xs2.T, g)
    return mix

def getSNR(signal, noise):
    """
    calcluate getSNR
    :param signal: signal
    :param noise: noise
    :return: SNR in log
    """
    return 20 * np.log10(norm(signal) / norm(noise))


def nextPow2(x):
    """
    calculate the nearest pow2 of x
    :param x:
    :return: the nearest pow2 of x
    """
    if x == 0:
        return 0
    else:
        return np.ceil(np.log2(x))



