import sdr
import matplotlib.pyplot as plt
import numpy as np


sdr.calculate_simulation_timing_parameters()


plt.figure(figsize=(4, 3))
x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
plt.subplot(121)
plt.grid()
plt.title("Sent Vectors")
plt.scatter(np.real(vectors), np.imag(vectors))
continuous_samples = sdr.tx(x)
sdr.rx(continuous_samples)
plt.subplot(122)
plt.title("Received Vectors")
sdr.plot_rx_constellation()


