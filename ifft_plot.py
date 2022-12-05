import sdr
import matplotlib.pyplot as plt
import numpy as np

x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)
plt.figure(figsize=(4, 3))
plt.subplot(221)
plt.title('Sub-Carrier Amplitude Real Part')
plt.stem(np.real(vectors))
plt.subplot(222)
plt.title('Digital Baseband Signal Real Part')
plt.stem(np.real(samples))
plt.subplot(223)
plt.title('Sub-Carrier Amplitude Imaginary Part')
plt.stem(np.imag(vectors))
plt.subplot(224)
plt.title('Digital Baseband Signal Imaginary Part')
plt.stem(np.imag(samples))
plt.show()
