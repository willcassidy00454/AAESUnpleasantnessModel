import Utils
import numpy as np
from matplotlib import pyplot as plt
import Energy


def showPlots(num_octave_bands, energy_spectra, energy_spectrum_freqs, octave_band_centres, flutter_score):
    fig, axes = plt.subplots(num_octave_bands)
    fig.set_size_inches(6, 8)
    fig.set_layout_engine("tight")
    plt.suptitle(f"|FFT(Energy Decay Fluctuations)| (flutter = {round(flutter_score, 3)})")

    for octave_band in range(num_octave_bands):
        energy_spectrum = energy_spectra[:, octave_band]
        axes[octave_band].plot(energy_spectrum_freqs, energy_spectrum)
        axes[octave_band].set_title(f"{octave_band_centres[octave_band]} Hz")

    plt.show()

def getFlutterEchoScore(rir, sample_rate, should_show_plots=False):
    # Truncate start of RIR up to max magnitude # # # currently truncates second half too
    rir = rir[np.argmax(abs(rir)):]

    # Get octave bands from RIR
    rir_octave_bands, octave_band_centres = Utils.getOctaveBandsFromIR(rir, sample_rate)
    num_octave_bands = rir_octave_bands.shape[1]

    # Get energy spectrum of each octave band, truncate between 1 and 20 Hz,
    # and calculate distance of max in each from mean, minus std dev
    fft_size = 2 ** 9
    etc_window_duration_ms = 10.0
    energy_spectrum_freqs = np.fft.rfftfreq(fft_size, etc_window_duration_ms / 1000.0)

    octave_band_scores = np.zeros(num_octave_bands)

    min_energy_freq_index = Utils.findIndexOfClosest(energy_spectrum_freqs, 1)
    max_energy_freq_index = Utils.findIndexOfClosest(energy_spectrum_freqs, 20)

    energy_spectra = np.zeros([max_energy_freq_index - min_energy_freq_index, num_octave_bands])

    for octave_band in range(num_octave_bands):
        energy_spectrum = Energy.getEnergySpectrum(rir_octave_bands[:, octave_band], sample_rate, fft_size, etc_window_duration_ms)
        energy_spectrum = energy_spectrum[min_energy_freq_index:max_energy_freq_index]
        energy_spectra[:, octave_band] = energy_spectrum
        energy_mean = np.mean(energy_spectrum)
        energy_stddev = np.std(energy_spectrum)
        energy_max = np.max(energy_spectrum)
        octave_band_scores[octave_band] = (energy_max - energy_mean) / energy_stddev

    flutter_echo_score = np.mean(octave_band_scores)

    if should_show_plots:
        showPlots(num_octave_bands, energy_spectra, energy_spectrum_freqs[min_energy_freq_index:max_energy_freq_index], octave_band_centres, flutter_echo_score)

    return flutter_echo_score