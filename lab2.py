import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

from scipy.signal.windows import hann, flattop

f1 = 300 # Hz
f2 = 300.25 # Hz
f3 = 299.75  # Hz
fs = 400 # Hz
N = 2000
k = np.arange(N)
xkMax = 2
x1 = xkMax * np.sin(2 * np.pi * f1 / fs * k)
x2 = xkMax * np.sin(2 * np.pi * f2 / fs * k)
x3 = xkMax * np.sin(2 * np.pi * f3 / fs * k)

wrect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)

X1wrect = fft(x1 * wrect)
X2wrect = fft(x2 * wrect)
X3wrect = fft(x3 * wrect)
X1whann = fft(x1 * whann)
X2whann = fft(x2 * whann)
X3whann = fft(x3 * whann)
X1wflattop = fft(x1 * wflattop)
X2wflattop = fft(x2 * wflattop)
X3wflattop = fft(x3 * wflattop)

# wykresy
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(np.abs(X1wrect), 'C0o-', ms=3, label='x1 rect')
plt.plot(np.abs(X2wrect), 'C1o-', ms=3, label='x2 rect')
plt.plot(np.abs(X3wrect), 'C2o-', ms=3, label='x3 rect')
plt.xlabel(r'$k$')
plt.ylabel(r'$|X[k]|$')
plt.title('DFT spectra - Rectangular Window')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(np.abs(X1whann), 'C0o-', ms=3, label='x1 hann')
plt.plot(np.abs(X2whann), 'C1o-', ms=3, label='x2 hann')
plt.plot(np.abs(X3whann), 'C2o-', ms=3, label='x3 hann')
plt.xlabel(r'$k$')
plt.ylabel(r'$|X[k]|$')
plt.title('DFT spectra - Hann Window')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(np.abs(X1wflattop), 'C0o-', ms=3, label='x1 flattop')
plt.plot(np.abs(X2wflattop), 'C1o-', ms=3, label='x2 flattop')
plt.plot(np.abs(X3wflattop), 'C2o-', ms=3, label='x3 flattop')
plt.xlabel(r'$k$')
plt.ylabel(r'$|X[k]|$')
plt.title('DFT spectra - Flattop Window')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
