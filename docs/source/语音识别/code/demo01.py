import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2, int(fft_size/2) + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()


# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # signal 是音频信号，sample_rate 是采样率。fft_size 是用于离散傅里叶变换（FFT）的窗口大小。 spec 是频谱数据
    sample_rate, signal = wavfile.read('./zh.wav')
    # 打印采样率和信号长度的信息。
    signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    print('sample rate:', sample_rate, ', frame length:', len(signal))
    # plot_time函数绘制时域图，显示音频信号的波形。
    # plot_freq 函数绘制频域图，显示音频信号的频谱信息。
    plot_time(signal, sample_rate)
    plot_freq(signal, sample_rate)

