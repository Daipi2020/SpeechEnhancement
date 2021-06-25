import basic_functions
import matplotlib.pyplot as plt
import librosa
noise_path = './white.wav'
basic_functions.addNoise('./clean_16000.wav', noise_path, 0, display=True)
basic_functions.addNoise('./newscut_tail_16000.wav', noise_path, 0, display=True)

origin_signal_path = './newscut_tail_16000.wav'

