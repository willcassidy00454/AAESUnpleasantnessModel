import numpy as np

def getEDC(rir, sample_rate):
    integration_limit_samples = len(rir)
    reversed_rir = rir[::-1]

    # Calculate Schroeder decay
    edc_dB_reversed = 10.0 * np.log10(np.cumsum(np.square(reversed_rir)) / np.sum(np.square(rir)))
    edc_dB = edc_dB_reversed[::-1]

    time_values_samples = range(integration_limit_samples)
    time_values_seconds = [time_value / sample_rate for time_value in time_values_samples]

    return edc_dB, time_values_seconds
