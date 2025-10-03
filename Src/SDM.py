import warnings

import numpy as np
from matplotlib import pyplot as plt
import Utils
from scipy.signal import savgol_filter
import Energy
from scipy.signal import butter, sosfilt


# spatial_ir: impulse response in B-format
def getDOAPerSample(spatial_ir, window_length_samples=5):
    # Get each axis as the product of the omni channel and each respective bidirectional channel, smoothed with hanning
    assert (window_length_samples >= 5)
    window = np.hanning(window_length_samples) # Note: this is slightly different to the MATLAB Hanning window
    coords_cartesian = np.zeros([spatial_ir.shape[0], 3])

    for axis in range(3):
        spherical_harmonic_index = axis + 1
        coords_cartesian[:, axis] = np.convolve(window, spatial_ir[:, 0] * spatial_ir[:, spherical_harmonic_index], "same")

    # Normalise each direction to a radius of 1
    euclidean_distances = np.tile(np.sqrt(np.square(coords_cartesian[:, 0])
                                          + np.square(coords_cartesian[:, 1])
                                          + np.square(coords_cartesian[:, 2])), (3, 1)).transpose()

    doa_per_sample_cartesian = coords_cartesian / euclidean_distances

    return doa_per_sample_cartesian


# Returns plot_angles_rad, radii_dB
def getSpatioTemporalMap(spatial_ir,
                         sample_rate,
                         start_ms=-1,
                         duration_ms=200,
                         start_is_relative_to_direct=True,
                         plane="transverse",
                         num_plot_angles=300):
    doa_cartesian = getDOAPerSample(spatial_ir)

    start_samples = int(np.floor(sample_rate * start_ms / 1000))
    duration_samples = int(np.floor(sample_rate * duration_ms / 1000))

    if start_is_relative_to_direct:
        # Assumes the direct sound arrives as the maximum sample of the omni channel
        direct_sample_index = np.argmax(np.abs(spatial_ir[:, 0]))

        # Set start relative to the direct sample
        start_index = np.max([0, direct_sample_index + start_samples])
        end_index = np.min([spatial_ir.shape[0] - 1, start_index + duration_samples])
    else:
        start_index = start_samples
        end_index = start_index + duration_samples

    # Truncate DOAs to time region
    doa_cartesian_trunc = doa_cartesian[start_index:end_index, :]

    # Transform cartesian coords and convert to spherical, where doa_spherical = (radii, azimuths, elevations)
    # Note: doa_spherical[:, 0] will be ignored as radius is taken from pressure
    if plane == "lateral":
        doa_spherical_rad = Utils.cartesianToSpherical(doa_cartesian_trunc)
    elif plane == "median":
        transformed_doa_cartesian = np.zeros_like(doa_cartesian_trunc)
        transformed_doa_cartesian[:, 0] = doa_cartesian_trunc[:, 0]
        transformed_doa_cartesian[:, 1] = doa_cartesian_trunc[:, 2]
        transformed_doa_cartesian[:, 2] = -doa_cartesian_trunc[:, 1]
        doa_spherical_rad = Utils.cartesianToSpherical(transformed_doa_cartesian)
    elif plane == "transverse":
        transformed_doa_cartesian = np.zeros_like(doa_cartesian_trunc)
        transformed_doa_cartesian[:, 0] = doa_cartesian_trunc[:, 2]
        transformed_doa_cartesian[:, 1] = doa_cartesian_trunc[:, 1]
        transformed_doa_cartesian[:, 2] = -doa_cartesian_trunc[:, 0]
        doa_spherical_rad = Utils.cartesianToSpherical(transformed_doa_cartesian)
        doa_spherical_rad[:, 1] += np.pi / 2
    else:
        warnings.warn("Plane argument not recognised (defaulting to 'lateral')")
        doa_spherical_rad = Utils.cartesianToSpherical(doa_cartesian_trunc)

    # Apply arbitrary offsets for alignment correction
    azimuth_offset_rad = np.pi
    elevation_offset_rad = 0
    doa_spherical_rad[:, 1] += azimuth_offset_rad
    doa_spherical_rad[:, 2] += elevation_offset_rad

    # Map (-pi to pi) to (0 to 1), preserving values outside range (these get wrapped in the next step)
    angles_0to1 = (doa_spherical_rad[:, 1] + np.pi) / (2 * np.pi)

    # Quantise to num_plot_angles, mapping to (0 to (num_plot_angles - 1)) with wrapping
    angles_0toN_quantised = np.round(angles_0to1 * num_plot_angles)
    angles_0toN_wrapped = angles_0toN_quantised % num_plot_angles

    angles_rad = np.linspace(-np.pi, np.pi - (2 * np.pi / num_plot_angles), num_plot_angles)
    radii = np.zeros(num_plot_angles)

    # Get energy from the omnidirectional rir channel (this is used for the radius)
    pressure = spatial_ir[start_index:end_index, 0]
    energy_linear = np.square(pressure)

    for angle_index in range(num_plot_angles):
        indices = angles_0toN_wrapped == angle_index
        radii[angle_index] = np.nansum(energy_linear[indices] * np.abs(np.cos(doa_spherical_rad[indices, 2])))

    window_length = 5
    radii_wrapped_for_start = radii[-window_length - 1:-1]
    radii_wrapped_for_end = radii[:window_length]
    radii_to_smooth = np.concat([radii_wrapped_for_start, radii, radii_wrapped_for_end])
    radii_smoothed = savgol_filter(radii_to_smooth, window_length, 1)
    radii_smoothed = radii_smoothed[window_length:-window_length]

    # Convert energy radius to decibels, clipping at -60 dB
    radii_dB = 10 * np.log10(np.clip(radii_smoothed, 1e-6, None))

    # Mirror along the x-axis to match Treble presentation
    angles_rad_corrected = np.pi - angles_rad

    return angles_rad_corrected, radii_dB


def plotSpatioTemporalMap(spatial_rir, sample_rate, plane="median", num_plot_angles=200):
    fig, axes = plt.subplots(3, 2, subplot_kw={'projection': 'polar'})

    starts_relative_to_direct_ms = [-1, 10, 100, 200, 400, 800]

    for index, duration_ms in enumerate([3,20,20,20,20,20]):
        angles_rad, radii_dB = getSpatioTemporalMap(spatial_rir,
                                                    sample_rate,
                                                    start_ms=starts_relative_to_direct_ms[index],
                                                    duration_ms=duration_ms,
                                                    start_is_relative_to_direct=True,
                                                    plane=plane,
                                                    num_plot_angles=num_plot_angles)

        axes[index % 3][int(index / 3)].fill(angles_rad, radii_dB, color="black", alpha=1 / (index + 1))
                 # label=f"{starts_relative_to_direct_ms[index]}-{starts_relative_to_direct_ms[index] + duration_ms}")
        axes[index % 3][int(index / 3)].set_axisbelow(True)
        # axes[index % 3][int(index / 3)].legend(title='Time Region (ms)', bbox_to_anchor=(1, 1))
        axes[index % 3][int(index / 3)].set_title(f"{starts_relative_to_direct_ms[index]}-{starts_relative_to_direct_ms[index] + duration_ms}ms", x=1.6, y=0.8)
    plt.show()


def getSpatialAsymmetryScore(spatial_rir, sample_rate, show_plots=False):
    hpf_cutoff_Hz = 500.0
    sos = butter(2, hpf_cutoff_Hz, 'highpass', fs=sample_rate, output='sos')
    hpf_omni_rir = sosfilt(sos, spatial_rir[:, 0])
    # # spatial_rir[:, 1] = sosfilt(sos, spatial_rir[:, 1])
    # # spatial_rir[:, 2] = sosfilt(sos, spatial_rir[:, 2])
    # # spatial_rir[:, 3] = sosfilt(sos, spatial_rir[:, 3])

    # Get EDC of omni component
    edc_dB, edc_times = Energy.getEDC(spatial_rir[:, 0], sample_rate)
    #
    # # Find time region between -30 and -40 dB decay (late energy)
    # late_start_samples = Utils.findIndexOfClosest(edc_dB, -15)
    # # late_end_samples = Utils.findIndexOfClosest(edc_dB, -35)
    #
    # late_start_ms = edc_times[late_start_samples] * 1000
    # # late_end_ms = edc_times[late_end_samples] * 1000

    rir_duration_ms = edc_times[Utils.findIndexOfClosest(edc_dB, -45)] * 1000
    # rir_duration_ms = 550
    first_order_rir = zeroPadOrTruncateToDuration(spatial_rir, sample_rate, rir_duration_ms)

    start_times_ms = [edc_times[Utils.findIndexOfClosest(edc_dB, -10)] * 1000,
                      edc_times[Utils.findIndexOfClosest(edc_dB, -20)] * 1000,
                      edc_times[Utils.findIndexOfClosest(edc_dB, -30)] * 1000,
                      edc_times[Utils.findIndexOfClosest(edc_dB, -40)] * 1000]
    # start_times_ms = [300.0]

    median_plane_scores = np.zeros_like(start_times_ms)
    transverse_plane_scores = np.zeros_like(start_times_ms)
    lateral_plane_scores = np.zeros_like(start_times_ms)

    for score_index, start_ms in enumerate(start_times_ms):
        median_plane_scores[score_index] = getAsymmetryScoreForTimeRegion(first_order_rir,
                                                                          sample_rate,
                                                                          start_ms,
                                                                          50,
                                                                          True,
                                                                          show_plots,
                                                                          "median")
        transverse_plane_scores[score_index] = getAsymmetryScoreForTimeRegion(first_order_rir,
                                                                          sample_rate,
                                                                          start_ms,
                                                                          10,
                                                                          True,
                                                                          show_plots,
                                                                          "transverse")
        lateral_plane_scores[score_index] = getAsymmetryScoreForTimeRegion(first_order_rir,
                                                                          sample_rate,
                                                                          start_ms,
                                                                          50,
                                                                          True,
                                                                          show_plots,
                                                                          "lateral")

    return np.mean(median_plane_scores) - np.max(lateral_plane_scores)# + np.max(lateral_plane_scores)


def getOffCentreRatio(radii_dB, angles_rad):
    # Convert radii (linear) and angles (rad) into cartesian coords, find geometric mean, convert back to polar
    points_cartesian = Utils.pol2cart(10 ** (radii_dB / 10), angles_rad)
    mean_of_points_cartesian = [np.mean(points_cartesian[0]), np.mean(points_cartesian[1])]
    magnitude_of_mean_linear, angle_of_mean_rad = Utils.cart2pol(mean_of_points_cartesian[0],
                                                                 mean_of_points_cartesian[1])

    # Off-centre factor is the magnitude of the centre of gravity (mean) of the peak-normalised late spatial energy
    # This differs from the mean of the radii as a near-zero mean could be yielded for a narrow but symmetrical response
    off_centre_ratio_dB = 10 * np.log10(magnitude_of_mean_linear)

    return off_centre_ratio_dB, angle_of_mean_rad


def getNarrowness(radii_dBFS):
    # Since the radii are peak-normalised, taking the mean radius represents the surface area
    # i.e. a circular response would yield the highest mean (unity), and a very narrow response would be near zero
    late_surface_area_linear = np.mean(10 ** (radii_dBFS / 10))
    narrowness = 1.0 - late_surface_area_linear

    return narrowness


def getAngleWeighting(angle_rad_unwrapped):
    # Wrap the mean angle of the late energy within 0-2pi
    # direct_angle_rad_wrapped = direct_angle_rad % (2.0 * np.pi)
    angle_rad_wrapped = angle_rad_unwrapped % (2.0 * np.pi)

    # Angle weighting is such that hard L/R is 1 and front/back is 0
    # angle_weighting = np.abs(np.sin(angle_rad_wrapped - (np.pi / 2))) # for max at 90 deg
    angle_weighting = (1.0 - np.cos(angle_rad_wrapped)) * 0.5 # for max at 180 deg

    return angle_weighting


# Determine how much the late energy is weighted towards one direction, and score more highly for angle regions defined by getAngleWeighting()
def getAsymmetryScoreForTimeRegion(spatial_rir, sample_rate, start_ms, duration_ms, start_is_relative_to_direct=True, show_plots=True, plane="median"):
    # Get DOA angles and radii for time region
    angles_rad, radii_dB = getSpatioTemporalMap(spatial_rir,
                                                sample_rate,
                                                start_ms=start_ms,
                                                duration_ms=duration_ms,
                                                start_is_relative_to_direct=start_is_relative_to_direct,
                                                plane=plane,
                                                num_plot_angles=100)

    # Normalise radii to 0 dBFS
    max_dB = np.max(radii_dB)
    radii_dBFS = radii_dB - max_dB

    off_centre_ratio_dB, angle_of_mean_rad = getOffCentreRatio(radii_dBFS, angles_rad)

    narrowness = getNarrowness(radii_dBFS)

    angle_weighting = getAngleWeighting(angle_of_mean_rad)

    # Spatial score is how off-centre the late energy is, where narrow responses contribute to a higher score
    # Late energy that is one-sided and lateral will increase the score
    spatial_score = narrowness #+ angle_weighting

    if show_plots:
        fig, axes = plt.subplots(subplot_kw={'projection': 'polar'})
        # axes.set_ylim([min_normalised_dB - 5, 0])
        # plt.plot([direct_angle_rad, direct_angle_rad], [-5, 0], color="blue", alpha=1, label="Direct")
        plt.plot([angle_of_mean_rad, angle_of_mean_rad], [-5, 0], color="black", alpha=1, label="Late")
        plt.fill(angles_rad, radii_dBFS, color="black", alpha=0.3, label=f"Late ({np.round(start_ms)}-{np.round(start_ms + duration_ms)} ms)")
        plt.scatter(angle_of_mean_rad, off_centre_ratio_dB, label="Mean of Late")
        axes.set_axisbelow(True)
        # plt.legend(loc='best')
        plt.suptitle(f"Spatial Asymmetry Score = {np.round(spatial_score, 2)}")
        plt.show()

    return spatial_score


def zeroPadOrTruncateToDuration(spatial_rir, sample_rate, duration_after_direct_ms):
    # This only returns the first four channels
    direct_position = np.argmax(np.abs(spatial_rir[:, 0]))

    overall_length_samples = int(np.floor(sample_rate * (duration_after_direct_ms / 1000.0) + direct_position))

    resized_first_order_rir = np.zeros([overall_length_samples, 4])

    for channel_index in range(4):
        num_samples_to_insert = np.min([len(spatial_rir[:, 0]), overall_length_samples])
        resized_first_order_rir[:, channel_index] = spatial_rir[:num_samples_to_insert, channel_index]

    return resized_first_order_rir