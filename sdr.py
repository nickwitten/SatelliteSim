import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz


NUM_SUB_CARRIERS = 8  # Number of sub-carriers in the OFDM channel
E_SQRT = 5  # Magnitude of constellation vectors
SCS = 60e3 # Sub-carrier spacing in Hz
CARRIER_FREQ = 60e6  # Carrier frequency in Hz
TS = 1 / (2 * SCS * NUM_SUB_CARRIERS)  # DAC/ADC sample rate
SIMULATION_TS = 1 / (16 * CARRIER_FREQ)
CARRIER_SAMPLES_I = np.array([np.cos(2 * np.pi * CARRIER_FREQ * t) for t in
    np.arange(0, NUM_SUB_CARRIERS * TS, SIMULATION_TS)])  # Simulation in-phase "continuous" samples
CARRIER_SAMPLES_Q = np.array([np.sin(2 * np.pi * CARRIER_FREQ * t) for t in
    np.arange(0, NUM_SUB_CARRIERS * TS, SIMULATION_TS)])  # Simulation quadrature "continuous" samples


def symbol_map_tx(data: bytes):
    """ Map 16 bit stream onto 8 complex vectors using QPSK bit mapping 
        Each byte is LSB first in the symbol array """
    assert len(data) == NUM_SUB_CARRIERS * 2 / 8
    symbols = np.zeros(8, dtype=np.complex128)
    # Loop through 64 bits
    for i in range(0, len(data) * 8, 2):
        byte_index = int(i / 8)
        symbol = complex(
            ((data[byte_index] >> (i % 8)) & 1) * 2 - 1,
            ((data[byte_index] >> (i % 8 + 1)) & 1) * 2 - 1
        )
        symbols[int(i / 2)] = symbol
    return symbols * E_SQRT

def symbol_map_rx(symbols: list):
    """ Return 16 bit stream from 8 complex vectors using QPSK bit mapping and
    least distance constellation decisions """
    assert len(symbols) == NUM_SUB_CARRIERS
    data = bytearray(int(NUM_SUB_CARRIERS * 2 / 8))
    # Loop through 32 symbols
    for i, symbol in enumerate(symbols):
        data[int(i / 4)] |= int(((np.real(symbols[i]) >= 0) + 1) / 2) << (i % 4 * 2)
        data[int(i / 4)] |= int(((np.imag(symbols[i]) >= 0) + 1) / 2) << (i % 4 * 2 + 1)
    return bytes(data)

def ifft_tx(symbols: list):
    """ Return an 8 point ifft to transform symbols from frequency domain to
    time domain samples """
    assert len(symbols) == NUM_SUB_CARRIERS
    return np.fft.ifft(symbols)

def fft_rx(samples: list):
    """ Return an 8 point fft to transform received time domain samples to
    frequency domain symbols """
    assert len(samples) == NUM_SUB_CARRIERS
    return np.fft.fft(samples)

def up_converter_tx(samples: list):
    """ Take in 8 complex time domain samples and modulate with the carrier
    frequency. Returns "continuous" time domain array """
    assert len(samples) == 8
    # Up sample baseband samples to simulate DAC conversion
    samples = np.expand_dims(samples, axis=1)
    up_sample_rate = int(TS / SIMULATION_TS)
    samples = np.tile(samples, (1, up_sample_rate)).flatten()
    # Mix with carrier frequency
    mixed_samples_i = np.real(samples) * CARRIER_SAMPLES_I
    mixed_samples_q = np.imag(samples) * CARRIER_SAMPLES_Q
    # Add the mixed IQ signals
    return mixed_samples_i + mixed_samples_q

def down_converter_rx(continuous: list):
    """ Take in continuous time domain array, demodulate to baseband, and
    sample. Returns 8 complex time domain samples """
    # Multiply by carrier frequency
    demodulated_i = continuous * CARRIER_SAMPLES_I
    demodulated_q = continuous * CARRIER_SAMPLES_Q
    # Convolve with low pass filter to remove double frequency and out of band noise
    b, a = scipy.signal.iirfilter(4, Wn=(1.5 * NUM_SUB_CARRIERS * SCS),
            fs=(1 /SIMULATION_TS), btype="low", ftype="butter")
    demodulated_i = lfilter(b, a, demodulated_i)
    demodulated_q = lfilter(b, a, demodulated_q)
    # Sample the signals to simulate ADC conversion
    sample_rate = int(TS / SIMULATION_TS)
    samples_i = demodulated_i[sample_rate - 1::sample_rate]
    samples_q = demodulated_q[sample_rate - 1::sample_rate]
    # Place back into complex values and mutliply by 2 because of amplitude
    # loss in demodulation
    return np.array([2 * complex(real, imag) for real, imag in zip(samples_i, samples_q)])


if __name__ == '__main__':
    data = "Hi".encode()
    print(f"Sending {data.decode()}")

    tx_symbols = symbol_map_tx(data)
    tx_baseband_samples = ifft_tx(tx_symbols)
    tx_continuous_samples = up_converter_tx(tx_baseband_samples)

    plt.plot(np.arange(0, tx_continuous_samples.size * SIMULATION_TS, SIMULATION_TS), tx_continuous_samples)
    plt.show()

    rx_baseband_samples = down_converter_rx(tx_continuous_samples)
    rx_vectors = fft_rx(rx_baseband_samples)
    data = symbol_map_rx(rx_vectors)

    print(f"Received {data.decode()}")
