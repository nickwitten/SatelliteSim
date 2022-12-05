import sdr
import matplotlib.pyplot as plt
import numpy as np


sdr.CARRIER_FREQ = 30e9  # Carrier frequency in Hz
sdr.SIMULATION_TS = 1 / (4 * sdr.CARRIER_FREQ)
sdr.calculate_simulation_timing_parameters()


x = bytes(b'\x1B\x1B')
vectors = sdr.symbol_map_tx(x)
samples = sdr.ifft_tx(vectors)
continuous_samples = sdr.up_converter_tx(samples)
samples = sdr.down_converter_rx(continuous_samples)
vectors = sdr.fft_rx(samples)

plt.figure(figsize=(4, 3))
plt.subplot(221)
plt.title("Inphase Time Sampled")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.stem(np.real(samples))
plt.subplot(222)
plt.title("Digital Frequency Real Part")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.stem(np.real(vectors))
plt.subplot(223)
plt.title("Quadrature Time Sampled")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.stem(np.imag(samples))
plt.subplot(224)
plt.title("Digital Frequency Imaginary Part")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.stem(np.imag(vectors))

plt.show()


