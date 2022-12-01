import sdr
import numpy as np
import matplotlib.pyplot as plt
import IPython

N0 = sdr.EB / 1000
C = 3e8
SIMULATION_TS_NO_DOPPLER = sdr.SIMULATION_TS  # Save time scale without doppler

def doppler_shift_on(samples, velocity):
    """ Change the simulation time base to add a doppler effect after a
    transmission. doppler_shift_off() will turn off the new time scale. """
    sdr.SIMULATION_TS -= velocity * sdr.SIMULATION_TS / C
    sdr.SIMULATION_TS = sdr.TS / (sdr.TS // sdr.SIMULATION_TS)  # Make sure sim fits integer number of times in ADC
    print(f"Changing time base by {sdr.SIMULATION_TS - SIMULATION_TS_NO_DOPPLER}")
    sdr.calculate_simulation_timing_parameters()
    num_blocks = round(samples.size / sdr.NUM_SIMULATION_SAMPLES_BLOCK)
    new_samples = np.empty(num_blocks * sdr.NUM_SIMULATION_SAMPLES_BLOCK)
    samples_off = samples.size - new_samples.size
    if samples_off < 0:
        new_samples[:samples.size] = samples[:]
        new_samples[samples.size:] = samples[-1]
    else:
        new_samples[:] = samples[:new_samples.size]
    return new_samples

def doppler_shift_off():
    sdr.SIMULATION_TS = SIMULATION_TS_NO_DOPPLER
    sdr.calculate_simulation_timing_parameters()

def attenuate(samples, db):
    return samples / (10 ** (db / 10))

def add_noise(samples, power):
    noise = np.random.normal(0, np.sqrt(power), continuous_samples.size)
    return samples + noise

if __name__ == '__main__':
    tx_data = "Hello World!".encode()
    print(f"Sending:\n\t{tx_data.decode()}")

    continuous_samples = sdr.tx(tx_data)

    continuous_samples = doppler_shift_on(continuous_samples, 2500)
    continuous_samples = attenuate(continuous_samples, 10)
    continuous_samples = add_noise(continuous_samples, N0 / 2)

    rx_data = sdr.rx(continuous_samples)
    sdr.plot_rx_constellation()

    print(f"Received:")
    try:
        print(f"\t{rx_data.decode()}")
    except:
        print("\tUnable to decode.")
        print(f"\t{tx_data.hex()}")
        print(f"\t{rx_data.hex()}")
        pass

