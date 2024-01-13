import numpy as np
import matplotlib.pyplot as plt

# ACF
def calculate_acf(signal):
    N = len(signal)
    acf = np.correlate(signal, signal, mode='full') / N
    return acf[N-1:]

# Function to calculate linear mean and variance
def calculate_mean_and_variance(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    return mean, variance

# PSD
def calculate_psd(signal, fs):
    f, psd = plt.psd(signal, NFFT=1024, Fs=fs)
    plt.close() 
    return f, psd

# Noise signal generation
np.random.seed(42)
N = 1000
x = np.random.normal(0, 1, N)

# Defining the transfer function of the LTI system
Omega_c = np.pi / 2
H = np.where(np.abs(np.fft.fftfreq(N)) < Omega_c, 1, 0)

# y[k]
y = np.fft.ifft(np.fft.fft(x) * H)

acf_y = calculate_acf(y)
mean_y, variance_y = calculate_mean_and_variance(y)

fs = 1 
frequencies, psd_y = calculate_psd(y, fs)

autocorr_y = calculate_acf(y)
h_hat = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y)))

# Show
print("Auto-Correlation Function of y[k]:", autocorr_y)
print("Linear Mean of y[k]:", mean_y)
print("Variance of y[k]:", variance_y)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(h_hat.real)
plt.title('Estimated Impulse Response hˆ[k]')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(frequencies, 10 * np.log10(psd_y))
plt.title('PSD of y[k]')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.subplot(3, 1, 3)
plt.plot(autocorr_y)
plt.title('Auto-Correlation Function of y[k]')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.tight_layout()
plt.show()