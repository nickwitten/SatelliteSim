import numpy as np

E_sqrt = 5


def symbol_map_tx(data: bytes):
    """ Map 64 bit stream onto 32 complex vectors using QPSK bit mapping 
        Each byte is LSB first in the symbol array """
    assert len(data) == 8
    symbols = np.zeros(32, dtype=np.complex128)
    # Loop through 64 bits
    for i in range(0, len(data) * 8, 2):
        byte_index = int(i / 8)
        symbol = complex(
            ((data[byte_index] >> (i % 8)) & 1) * 2 - 1,
            ((data[byte_index] >> (i % 8 + 1)) & 1) * 2 - 1
        )
        symbols[int(i / 2)] = symbol
    return symbols * E_sqrt

def symbol_map_rx(symbols: list):
    """ Return 64 bit stream from 32 complex vectors using QPSK bit mapping and
    least distance constellation decisions """
    assert len(symbols) == 32
    data = bytearray(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    # Loop through 32 symbols
    for i, symbol in enumerate(symbols):
        data[int(i / 4)] |= int(((np.real(symbols[i]) >= 0) + 1) / 2) << (i % 4 * 2)
        data[int(i / 4)] |= int(((np.imag(symbols[i]) >= 0) + 1) / 2) << (i % 4 * 2 + 1)
    return bytes(data)

def ifft_tx(symbols: list):
    """ Return a 32 point ifft to transform symbols from frequency domain to
    time domain samples """
    assert len(symbols) == 32
    return np.fft.ifft(symbols)

def fft_rx(samples: list):
    """ Return a 32 point fft to transform received time domain samples to
    frequency domain symbols """
    assert len(samples) == 32
    return np.fft.fft(samples)


if __name__ == '__main__':
    data = "ABCDEFGH".encode()

    tx_symbols = symbol_map_tx(data)
    print(tx_symbols)
    tx_time_samples = ifft_tx(tx_symbols)

    rx_vectors = fft_rx(tx_time_samples)
    data = symbol_map_rx(rx_vectors)

    print(data.decode())

