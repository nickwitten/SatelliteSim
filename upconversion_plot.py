import sdr
import matplotlib.pyplot as plt
import numpy as np


sdr.CARRIER_FREQ = 30e4  # Carrier frequency in Hz
sdr.SIMULATION_TS = 1 / (8 * sdr.CARRIER_FREQ)
sdr.calculate_simulation_timing_parameters()


x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)

plt.figure(figsize=(12, 9))
sdr.up_converter_tx(samples, plot=True)
plt.subplot(221)
plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,-3))
plt.subplot(222)
plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,-3))
plt.subplot(223)
plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,-3))
plt.subplot(224)
plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,-3))
plt.show()


