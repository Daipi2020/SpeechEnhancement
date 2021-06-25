import librosa
import matplotlib.pyplot as plt
import librosa.display

audio_path = 'clean8k16bit.wav'
x, sr = librosa.load(audio_path, sr=None)
print('数据x类型和采样率sr类型', type(x), type(sr))
print('数据x尺寸和采样率', x.shape, sr)

# 2.绘制波形
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, x_axis='time')
plt.savefig('波形0.svg', format='svg')  # 矢量图格式 svg

# 3.声谱图
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.savefig('声谱图0.svg', format='svg')

# 4.声谱图，频率轴转换为对数轴
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.savefig('声谱图-对数轴0.svg', format='svg')
