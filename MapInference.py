"""
@FileName: Inference.py
@Description: Implement Inference
@Author: Ryuk
@CreateDate: 2020/05/08
@LastEditTime: 2020/05/08
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from librosa import display
from basic_functions import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import time

def show(data, s):
    plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)    # 创建子图 ax1，该子图是2行一列的总图中的 第一个图
    ax2 = plt.subplot(2, 1, 2)    # 创建子图 ax2，该子图是2行一列的总图中的 第二个图
    plt.sca(ax1)                  # 选择子图ax1
    plt.plot(data)                # 在子图ax1 中绘制 data
    plt.sca(ax2)                  # 选择子图ax2
    plt.plot(s)                   # 在子图ax2 中绘制 s
    ax1.set_title('test-wave')    # 设置图体，plt.title
    ax1.set_xlabel('time')        # 设置x轴名称,plt.xlabel
    ax1.set_ylabel('amplitude')
    ax2.set_title('test_denoise_wave')  # 设置图体，plt.title
    ax2.set_xlabel('time')         # 设置x轴名称,plt.xlabel
    ax2.set_ylabel('amplitude')
    plt.show()

test_audio_path = './test.wav'

model = load_model("./model.h5")


data, fs = librosa.load(test_audio_path, sr=None)
data_duration = librosa.get_duration(data, fs)

win_length = 256
hop_length = 128
nfft = 512

spectrum = librosa.stft(data, win_length=win_length, hop_length=hop_length, n_fft=nfft)
magnitude = np.abs(spectrum).T
phase = np.angle(spectrum).T

#帧数
frame_num = magnitude.shape[0] - 4     #它的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度。
feature = np.zeros([frame_num, 257 * 5])
k = 0
for i in range(frame_num - 4):
    frame = magnitude[k:k + 5]
    feature[i] = np.reshape(frame, 257 * 5)
    k += 1

ss = StandardScaler()
feature = ss.fit_transform(feature)
mask = model.predict(feature)

fig = plt.figure(figsize=(14, 5))
plt.imshow(mask, cmap='Blues', interpolation='none')  #https://blog.csdn.net/a892573486/article/details/107542839
plt.show()

magnitude = magnitude[2:-2]
en_magnitude = np.multiply(magnitude, mask)
phase = phase[2:-2]

en_spectrum = en_magnitude.T * np.exp(1.0j * phase.T)
frame = librosa.istft(en_spectrum, win_length=win_length, hop_length=hop_length)
#print(type(en_spectrum), type(frame))

show(data, frame)

#输出去噪后的音频文件
librosa.output.write_wav('test_output.wav', frame, sr=fs)
time.sleep(2)
pesqTest('clean8k16bit.wav', 'noisy8k16bit.wav', 8000)



