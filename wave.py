import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
fs = 16000  # 画图时，数据的采样率

# 读取音频数据
wav_data, sr_ori = librosa.load("./tail_16000.wav", sr=fs, mono=True)  #sr_ori : 音频原始采样率

# ########### 画图
plt.subplot(4, 1, 1)
plt.title("波形图", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)

plt.plot(time, wav_data)
plt.xlim(0, max(time))
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

plt.subplot(4, 1, 2)
plt.title("语谱图", fontsize=15)
plt.specgram(wav_data, Fs=fs, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(4, 1, 3)
plt.title("语谱图", fontsize=15)
X = librosa.stft(wav_data)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=fs, x_axis='time', y_axis='hz')
#plt.colorbar()
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(4, 1, 4)
plt.title("语谱图", fontsize=15)
X = librosa.stft(wav_data)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=fs, x_axis='time', y_axis='log')
#plt.colorbar()
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.tight_layout()
plt.show()