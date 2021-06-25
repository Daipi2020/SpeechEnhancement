"""
@FileName: Mapping.py
@Description: Implement Mapping
@Author: Ryuk
@CreateDate: 2020/05/03
@LastEditTime: 2020/05/03
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import model_from_json
#adam = Adam(lr=1e-2)

adam = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)

def generateDataset():
    mix, sr = librosa.load("./noisy8k16bit.wav", sr=None)
    clean, sr = librosa.load("./clean8k16bit.wav",  sr=None)

    #判断mix和clean的数据长度大小，并修剪为相同长度
    len_mix = len(mix)
    len_clean = len(clean)
    if len_mix > len_clean:
        mix = mix[0:len_clean]
    elif len_mix < len_clean:
        clean = clean[0:len_mix]
    else:
        mix = mix
        clean = clean

    win_length = 256
    hop_length = 128
    nfft = 512

    mix_spectrum = librosa.stft(mix, win_length=win_length, hop_length=hop_length, n_fft=nfft)
    clean_spectrum = librosa.stft(clean, win_length=win_length, hop_length=hop_length, n_fft=nfft)

    mix_mag = np.abs(mix_spectrum).T
    clean_mag = np.abs(clean_spectrum).T

    # 调整输入为5帧
    frame_num = mix_mag.shape[0] - 4  # shape()的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度。
    feature = np.zeros([frame_num, 257*5])
    k = 0
    for i in range(frame_num):
        frame = mix_mag[k:k+5]
        feature[i] = np.reshape(frame, 257*5)
        k += 1

    label = clean_mag[2:-2]

    ss = StandardScaler()
    feature = ss.fit_transform(feature)
    return feature, label

def getModel():
    model = Sequential()
    model.add(Dense(2048, input_dim=1285))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(257))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model

def train(feature, label, model):

    # 定义优化器，loss function，训练过程中计算准确率
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    # 训练模型
    model.fit(feature, label, batch_size=128, epochs=200, validation_split=0.2)
    # 保存参数，载入参数
    #model.save_weights('my_model_weights.h5')
    #model.load_weights('my_model_weights.h5')
    # 保存网络结构，载入网络结构
    #json_string = model.to_json()
    #model = model_from_json(json_string)
    #print(json_string)
    model.save("./model.h5")

# 初始化
def main():
    feature, label = generateDataset()
    model = getModel() #从零开始创建模型进行训练
    #model = load_model('./model.h5')  #从现有模型model.h5开始训练
    ss = StandardScaler()
    feature = ss.fit_transform(feature)
    train(feature, label, model)

if __name__ == "__main__":
    main()





