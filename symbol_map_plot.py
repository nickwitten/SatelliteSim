import sdr
import matplotlib.pyplot as plt
import numpy as np

x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)
plt.figure(figsize=(4, 2))
plt.subplot(211)
plt.tight_layout()
plt.xlabel('Symbol Number')
plt.ylabel('Amplitude')
plt.title('Symbols Real Part')
plt.stem(np.real(vectors))
plt.subplot(212)
plt.xlabel('Symbol Number')
plt.ylabel('Amplitude')
plt.title('Symbols Imaginary Part')
plt.stem(np.imag(vectors))
plt.show()
