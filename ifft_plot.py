import sdr
import matplotlib.pyplot as plt
import numpy as np

x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)
plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.stem(np.real(samples))
plt.title('real part')
plt.grid()
plt.subplot(212)
plt.stem(np.imag(samples), markerfmt='C1o')
plt.title('imaginary part')
plt.grid()
plt.legend()
plt.show()
