import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz


NUM_SUB_CARRIERS = 8  # Number of sub-carriers in the OFDM channel
NUM_BITS_BLOCK = NUM_SUB_CARRIERS * 2  # Number of bits per block
EB = 10  # Energy per bit
SCS = 60e3 # Sub-carrier spacing in Hz
CARRIER_FREQ = 30e9  # Carrier frequency in Hz
# TS = 1 / (2 * SCS * NUM_SUB_CARRIERS)  # DAC/ADC sample rate
TS = 1 / SCS  # DAC/ADC sample rate
SIMULATION_TS = 1 / (4 * CARRIER_FREQ)
RX_CONSTELLATION_BUFFER = []  # Used to hold the latest received symbol vectors for plotting
NUM_SIMULATION_SAMPLES_BLOCK = None
CARRIER_SAMPLES_I = None
CARRIER_SAMPLES_Q = None


def calculate_simulation_timing_parameters():
    """ Recalculate simulation timing parameters """
    global NUM_SIMULATION_SAMPLES_BLOCK
    global CARRIER_SAMPLES_I
    global CARRIER_SAMPLES_Q
    NUM_SIMULATION_SAMPLES_BLOCK = NUM_SUB_CARRIERS * int(TS / SIMULATION_TS)  # Number of "continous" samples per block
    CARRIER_SAMPLES_I = np.array([np.cos(2 * np.pi * CARRIER_FREQ * n * SIMULATION_TS) for n in
        range(NUM_SIMULATION_SAMPLES_BLOCK)], dtype=np.float64)  # Simulation in-phase "continuous" samples
    CARRIER_SAMPLES_Q = np.array([np.sin(2 * np.pi * CARRIER_FREQ * n * SIMULATION_TS) for n in
        range(NUM_SIMULATION_SAMPLES_BLOCK)], dtype=np.float64)  # Simulation quadrature "continuous" samples
calculate_simulation_timing_parameters()

def tx(data: bytes):
    """ Simulate SDR transmission path from binary data to a continuous time
    signal. Must pass in an integer number of blocks of data bits. Returns an
    array of "continuous" samples that are ready to be transmitted into a
    channel. """
    # Calculate the number of blocks to transmit
    num_blocks = len(data) * 8 / (NUM_SUB_CARRIERS * 2)
    # Check that there is an integer number of blocks to transmit
    assert int(num_blocks) == num_blocks
    num_blocks = int(num_blocks)
    # Allocate a "continuous" time array
    tx_continuous_samples = np.zeros(num_blocks * NUM_SIMULATION_SAMPLES_BLOCK)
    # Loop through the blocks and apply all SDR operations
    for i in range(num_blocks):
        tx_symbols = symbol_map_tx(
            data[i*int(NUM_BITS_BLOCK / 8): \
                 (i+1)*int(NUM_BITS_BLOCK / 8)]
        )
        tx_baseband_samples = ifft_tx(tx_symbols)
        tx_continuous_samples[i * NUM_SIMULATION_SAMPLES_BLOCK: \
                              (i+1) * NUM_SIMULATION_SAMPLES_BLOCK] = \
            up_converter_tx(tx_baseband_samples)
    return tx_continuous_samples

def rx(continuous_samples: list):
    """ Simulate SDR receive path from a continuous time signal to binary data.
    Must pass in "continuous" time samples of the received signal. Returns an
    array of binary data. """
    global RX_CONSTELLATION_BUFFER
    # Calculate the number of blocks received
    num_blocks = continuous_samples.size / NUM_SIMULATION_SAMPLES_BLOCK
    # Check that there is an integer number of blocks received
    assert int(num_blocks) == num_blocks
    num_blocks = int(num_blocks)
    # Allocate binary data array
    rx_data = bytearray()
    # Reset the constellation buffer
    RX_CONSTELLATION_BUFFER = []
    # Loop through the blocks and apply all SDR operations
    for i in range(num_blocks):
        rx_baseband_samples = down_converter_rx(
            continuous_samples[i * NUM_SIMULATION_SAMPLES_BLOCK: \
                               (i+1) * NUM_SIMULATION_SAMPLES_BLOCK]
        )
        rx_vectors = fft_rx(rx_baseband_samples)
        rx_data.extend(symbol_map_rx(rx_vectors))
        RX_CONSTELLATION_BUFFER.append(rx_vectors)
    return bytes(rx_data)

def plot_rx_constellation():
    """ Plot the latest received signal vectors. Must be called after rx() """
    assert len(RX_CONSTELLATION_BUFFER) != 0
    for array in RX_CONSTELLATION_BUFFER:
        plt.scatter(np.real(array), np.imag(array))
    plt.grid()
    plt.show()

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
    return symbols * np.sqrt(EB)

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
    b, a = butter(3, 6 * NUM_SUB_CARRIERS * SCS, fs=(1/SIMULATION_TS), btype='low', analog=False)
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
    tx_data = "Hello World!".encode()
    print(f"Sending {tx_data.decode()}")

    continuous_samples = tx(tx_data)
    rx_data = rx(continuous_samples)

    plot_rx_constellation()

    try:
        print(f"Received {rx_data.decode()}")
    except:
        pass


