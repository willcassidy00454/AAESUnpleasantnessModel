import numpy as np
import Energy
import Utils
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import RT

def showPlots(early_mag_spectrum_log_smoothed, late_mag_spectrum_log_smoothed, frequencies, early_energy, late_energy, spectral_evolution_score):
    plt.figure()
    fig, axes = plt.subplots(1)
    plt.semilogx(frequencies, early_mag_spectrum_log_smoothed, label="Early", linestyle="--")
    plt.semilogx(frequencies, late_mag_spectrum_log_smoothed, label="Late", linestyle="-.")
    plt.legend()
    axes.set_xlabel("Frequency")
    fig.suptitle(f"Early = {np.round(early_energy, 2)}, Late = {np.round(late_energy, 2)}, Spectral Evolution = {np.round(spectral_evolution_score, 2)} dB")
    plt.show()

def getEarlyAndLateRIR(rir, sample_rate, early_start_dB, early_end_dB, late_start_dB, late_end_dB):
    edc_dB, _ = Energy.getEDC(rir, sample_rate)
    early_start_samples = Utils.findIndexOfClosest(edc_dB, early_start_dB)
    early_end_samples = Utils.findIndexOfClosest(edc_dB, early_end_dB)
    late_start_samples = Utils.findIndexOfClosest(edc_dB, late_start_dB)
    late_end_samples = Utils.findIndexOfClosest(edc_dB, late_end_dB)

    early_rir = rir[early_start_samples:early_end_samples]
    late_rir = rir[late_start_samples:late_end_samples]

    return early_rir, late_rir


def getSpectralEvolutionScore(rir, sample_rate, should_show_plots=False):
    rt = RT.estimateRT(rir, sample_rate)

    # Split early and late regions of the RIR
    early_rir, late_rir = getEarlyAndLateRIR(rir, sample_rate, -1, -15, -35, -40)

    # Zero-pad to the same length
    pad_length = np.max([len(early_rir), len(late_rir)])
    early_rir = np.pad(early_rir, (0, pad_length - len(early_rir)), mode='constant')
    late_rir = np.pad(late_rir, (0, pad_length - len(late_rir)), mode='constant')

    # Get magnitude spectrum of each
    early_mag_spectrum = 10 * np.log10(np.abs(np.fft.rfft(early_rir)))
    late_mag_spectrum = 10 * np.log10(np.abs(np.fft.rfft(late_rir)))

    # Convert to log frequency from cutoff to Nyquist
    cutoff = 2000
    early_mag_spectrum_log, early_frequencies = Utils.linearToLog(early_mag_spectrum, sample_rate, cutoff, sample_rate / 2)
    late_mag_spectrum_log, late_frequencies = Utils.linearToLog(late_mag_spectrum, sample_rate, cutoff, sample_rate / 2)

    # Smooth spectra
    smoothing_window_length_samples = early_mag_spectrum_log.shape[0] // 2
    early_mag_spectrum_log_smoothed = savgol_filter(early_mag_spectrum_log, window_length=smoothing_window_length_samples, polyorder=1)
    late_mag_spectrum_log_smoothed = savgol_filter(late_mag_spectrum_log, window_length=smoothing_window_length_samples, polyorder=1)

    # Normalise both spectra so they overlap (compensate for the overall decay in level, maybe use the RT and the time that has passed?)    early_max =
    early_mag_spectrum_log_smoothed -= np.max(early_mag_spectrum_log_smoothed)
    late_mag_spectrum_log_smoothed -= np.max(late_mag_spectrum_log_smoothed)

    # Compare energy of early and late magnitudes
    early_energy = np.mean(early_mag_spectrum_log_smoothed)
    late_energy = np.mean(late_mag_spectrum_log_smoothed)

    # # For each octave band, compute delta magnitude
    # deltas = late_mag_spectrum_log_smoothed - early_mag_spectrum_log_smoothed
    #
    # num_deltas = 10
    # deltas_downsampled = np.zeros([num_deltas, 2]) # [deltas, frequencies]
    # step = int(np.ceil(len(deltas) / num_deltas))
    #
    # for delta_index, sample_index in enumerate(range(0, len(deltas), step)):
    #     mean_delta = np.mean(deltas[sample_index:sample_index + step])
    #     deltas_downsampled[delta_index] = mean_delta, early_frequencies[sample_index]
    #
    # deltas_downsampled[:, 0] -= np.max(deltas_downsampled[:, 0])

    # Subtract deltas corresponding to a natural-sounding room
    # natural_deltas = [0.0,
    #                   -2.697296905008632,
    #                   -4.994258825874791,
    #                   -6.3869112057548305,
    #                   -6.639972913380748,
    #                   -6.706402403095581,
    #                   -7.304174210500316,
    #                   -9.432400673240224,
    #                   -10.769930530190639,
    #                   -12.260740071765039]
    # deltas_downsampled[:, 0] -= natural_deltas

    # Either sum scores or return all as list
    spectral_evolution_score = late_energy - early_energy

    spectral_evolution_score = (spectral_evolution_score + 12) / 14

    if should_show_plots:
        showPlots(early_mag_spectrum_log_smoothed, late_mag_spectrum_log_smoothed, early_frequencies, early_energy, late_energy, spectral_evolution_score)

    return spectral_evolution_score