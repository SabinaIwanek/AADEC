import numpy as np
import matplotlib.pyplot as plt

def generate_W_matrix(N):
    return np.fft.ifft(np.eye(N))

def generate_K_matrix(N):
    return np.diag(np.arange(N))

# Input data
x_mu = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=np.complex_)

# Length of the signal
N = len(x_mu)

# matrix W
W = generate_W_matrix(N)

# matrix K
K = generate_K_matrix(N)

# Signal synthesis
x = (1/N) * np.conjugate(W).T @ x_mu

# Plotting the signals
plt.stem(np.real(x_mu), label='Original Signal')
plt.stem(np.real(x), label='Synthesized Signal', markerfmt='rx')
plt.legend()
plt.title('Signal Synthesis using IDFT')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()
