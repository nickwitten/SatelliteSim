import sdr
import matplotlib.pyplot as plt
import numpy as np

x0 = bytes(b'\x00\x00')
x1 = bytes(b'\x55\x55')
x2 = bytes(b'\xAA\xAA')
x3 = bytes(b'\xFF\xFF')
vectors = sdr.symbol_map_tx(x0)
plt.scatter(np.real(vectors), np.imag(vectors), label='00')
vectors = sdr.symbol_map_tx(x1)
plt.scatter(np.real(vectors), np.imag(vectors), label='01')
vectors = sdr.symbol_map_tx(x2)
plt.scatter(np.real(vectors), np.imag(vectors), label='10')
vectors = sdr.symbol_map_tx(x3)
plt.scatter(np.real(vectors), np.imag(vectors), label='11')
plt.grid()
plt.legend()
plt.show()
