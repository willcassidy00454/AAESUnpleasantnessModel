import numpy as np
import Energy
import Utils
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

def showPlots(edc_dB,
              edc_times,
              early_start_dB,
              early_start_time,
              early_end_dB,
              early_end_time,
              late_start_dB,
              late_start_time,
              late_end_dB,
              late_end_time,
              curvature):
    plt.plot(edc_times, edc_dB)
    plt.plot([early_start_time, early_end_time], [early_start_dB, early_end_dB], 'bo-')
    plt.plot([late_start_time, late_end_time], [late_start_dB, late_end_dB], 'ro--')
    plt.show()

def getCurvature(rir, sample_rate, centre_freq=False, show_plots=False):
    if centre_freq:
        centre_freq = 1000.0

        centre_to_crossover_factor = 2.0 ** 0.5
        bin_lower = centre_freq / centre_to_crossover_factor
        bin_upper = centre_freq * centre_to_crossover_factor
        sos = butter(4, [bin_lower, bin_upper], 'bandpass', fs=sample_rate, output='sos')

        rir = sosfilt(sos, rir)

    edc_dB, edc_times = Energy.getEDC(rir, sample_rate)

    early_start_dB = -5.0
    early_end_dB = -10.0
    late_start_dB = -35.0
    late_end_dB = -40.0

    early_start_time = edc_times[Utils.findIndexOfClosest(edc_dB, early_start_dB)]
    early_end_time = edc_times[Utils.findIndexOfClosest(edc_dB, early_end_dB)]
    late_start_time = edc_times[Utils.findIndexOfClosest(edc_dB, late_start_dB)]
    late_end_time = edc_times[Utils.findIndexOfClosest(edc_dB, late_end_dB)]

    early_gradient = (early_end_dB - early_start_dB) / (early_end_time - early_start_time)
    late_gradient = (late_end_dB - late_start_dB) / (late_end_time - late_start_time)

    curvature = (late_gradient / early_gradient) - 1.0

    if show_plots:
        showPlots(edc_dB,
                  edc_times,
                  early_start_dB,
                  early_start_time,
                  early_end_dB,
                  early_end_time,
                  late_start_dB,
                  late_start_time,
                  late_end_dB,
                  late_end_time,
                  curvature)

    return curvature
