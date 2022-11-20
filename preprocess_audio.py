import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "data/blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # signal = sr * T -> 22050 * 30 ;; sr -> sample rate
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# fft -> spectrum ;; move from time domain to frequency domain
fft = np.fft.fft(signal)  # numpy 1D array, len(values) = total num of sample aka 22050 * 30 ;; complex vals

magnitude = np.abs(fft)  # abs value of complex vals... magnitude indicates contribution of each frequency to sound
frequency = np.linspace(0, sr, len(magnitude))  # numbers from 0hz to sample rate

# only really need to focus on left have since fft is symmetrical
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# stft -> spectrogram

n_fft = 2048  # number of samples per fft
hop_length = 512  # number of samples shifting

stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)  # complex num to magnitude

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)  # n_mfcc = num of coef 13-28
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
