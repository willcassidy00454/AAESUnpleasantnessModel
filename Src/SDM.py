import warnings

import numpy as np
from matplotlib import pyplot as plt
import Utils
from scipy.signal import savgol_filter
import Energy


# spatial_ir: impulse response in B-format
def getDOAPerSample(spatial_ir, window_length_samples=5):
    # Get each axis as the product of the omni channel and each respective bidirectional channel, smoothed with hanning
    assert (window_length_samples >= 5)
    window = np.hanning(window_length_samples) # Note: this is slightly different to the MATLAB Hanning window
    coords_cartesian = np.zeros([spatial_ir.shape[0] + window_length_samples - 1, 3])

    for axis in range(3):
        spherical_harmonic_index = axis + 1
        coords_cartesian[:, axis] = np.convolve(window, spatial_ir[:, 0] * spatial_ir[:, spherical_harmonic_index], "full")

    # Normalise each direction to a radius of 1
    euclidean_distances = np.tile(np.sqrt(np.square(coords_cartesian[:, 0])
                                          + np.square(coords_cartesian[:, 1])
                                          + np.square(coords_cartesian[:, 2])), (3, 1)).transpose()
    doa_per_sample_cartesian = coords_cartesian / euclidean_distances

    half_window_length_samples = int(np.floor(window_length_samples / 2))

    # Truncate
    doa_per_sample_cartesian = doa_per_sample_cartesian[half_window_length_samples:-half_window_length_samples, :]

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
        direct_sample_index = np.argmax(spatial_ir[:, 0])

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

    window_length = 3
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
    # Get EDC of omni component
    edc_dB, edc_times = Energy.getEDC(spatial_rir[:, 0], sample_rate)

    # Find time region between -30 and -50 dB decay (late energy)
    minus_30_position_samples = Utils.findIndexOfClosest(edc_dB, -30)
    minus_50_position_samples = Utils.findIndexOfClosest(edc_dB, -50)

    start_ms = edc_times[minus_30_position_samples] * 1000
    end_ms = edc_times[minus_50_position_samples] * 1000

    # Get late energy spatial bins (around the azimuth)
    late_angles_rad, radii_late_dB = getSpatioTemporalMap(spatial_rir,
                                                     sample_rate,
                                                     start_ms=start_ms,
                                                     duration_ms=end_ms - start_ms,
                                                     start_is_relative_to_direct=False,
                                                     plane="median",
                                                     num_plot_angles=100)

    # Get direct energy direction (argmax of a fairly high-res map between -1 and 3 ms relative to the direct)
    direct_angles_rad, radii_direct_dB = getSpatioTemporalMap(spatial_rir,
                                                     sample_rate,
                                                     start_ms=-1,
                                                     duration_ms=3,
                                                     start_is_relative_to_direct=True,
                                                     plane="median",
                                                     num_plot_angles=100)

    # Normalise radii to 0 dBFS
    max_dB = np.max(radii_late_dB)
    radii_late_dB -= max_dB
    radii_direct_dB -= max_dB

    # min_normalised_dB = np.min(radii_late_dB)

    direct_index = np.argmax(radii_direct_dB)
    direct_angle_rad = direct_angles_rad[direct_index]

    # General Process:
    # Determine how much the late energy is weighted towards one direction, and score more highly for lateral directions

    # Convert linear radii and angles into cartesian coords, find geometric mean, convert back to polar
    late_points_cartesian = Utils.pol2cart(10 ** (radii_late_dB / 10), late_angles_rad)
    late_points_cartesian_mean = [np.mean(late_points_cartesian[0]), np.mean(late_points_cartesian[1])]
    magnitude_of_late_mean_linear, angle_of_late_mean_rad = Utils.cart2pol(late_points_cartesian_mean[0], late_points_cartesian_mean[1])

    # Since the radii are peak-normalised, taking the mean radius represents the surface area
    # i.e. a circular response would yield the highest mean (unity), and a very narrow response would be near zero
    surface_area_dB = np.mean(radii_late_dB)

    # Wrap the mean angle of the late energy within 0-2pi
    # direct_angle_rad_wrapped = direct_angle_rad % (2.0 * np.pi)
    angle_of_late_mean_rad_wrapped = angle_of_late_mean_rad % (2.0 * np.pi)
    # absolute_angle_difference_rad = np.abs(direct_angle_rad_wrapped - angle_of_late_mean_rad_wrapped)
    # direct_late_angle_difference_rad = np.min([absolute_angle_difference_rad, (2.0 * np.pi) - absolute_angle_difference_rad])

    # Angle weighting is such that hard L/R is 1 and front/back is 0
    angle_weighting = abs(np.sin(angle_of_late_mean_rad_wrapped - (np.pi / 2)))

    # Off-centre factor is the magnitude of the centre of gravity (mean) of the peak-normalised late spatial energy
    # This differs from the mean of the radii as a near-zero mean could be yielded for a narrow but symmetrical response
    off_centre_factor_dB = 10 * np.log10(magnitude_of_late_mean_linear)

    # Spatial score is how off-centre the late energy is, where narrow responses contribute to a higher score
    # Late energy that is one-sided and lateral will increase the score
    spatial_score = (off_centre_factor_dB - surface_area_dB) * (angle_weighting + 1)

    if show_plots:
        fig, axes = plt.subplots(subplot_kw={'projection': 'polar'})
        # axes.set_ylim([min_normalised_dB - 5, 0])
        plt.plot([direct_angle_rad, direct_angle_rad], [-5, 0], color="blue", alpha=1, label="Direct")
        plt.plot([angle_of_late_mean_rad, angle_of_late_mean_rad], [-5, 0], color="black", alpha=1, label="Late")
        plt.fill(late_angles_rad, radii_late_dB, color="black", alpha=0.3, label=f"Late ({np.round(start_ms)}-{np.round(end_ms)} ms)")
        plt.scatter(angle_of_late_mean_rad, 10 * np.log10(magnitude_of_late_mean_linear), label="Mean of Late")
        axes.set_axisbelow(True)
        # plt.legend(loc='best')
        plt.suptitle(f"Spatial Asymmetry Score = {np.round(spatial_score, 2)}")
        plt.show()

    return spatial_score