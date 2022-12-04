import sdr
import matplotlib.pyplot as plt
import numpy as np


sdr.CARRIER_FREQ = 30e9  # Carrier frequency in Hz
sdr.SIMULATION_TS = 1 / (6 * sdr.CARRIER_FREQ)
sdr.calculate_simulation_timing_parameters()


x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)
continuous_samples = sdr.up_converter_tx(samples)
plt.figure(figsize=(12, 9))
samples = sdr.down_converter_rx(continuous_samples, plot=True)

plt.subplot(221)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.subplot(222)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.subplot(223)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.subplot(224)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.show()


