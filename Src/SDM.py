import warnings

import numpy as np
from matplotlib import pyplot as plt
import Utils
from scipy.signal import savgol_filter


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
                          start_relative_to_direct_ms=-1,
                          duration_ms=200,
                          plane="transverse",
                          num_plot_angles=300):
    doa_cartesian = getDOAPerSample(spatial_ir)

    # Assumes the direct sound arrives as the maximum sample of the omni channel
    direct_sample_index = np.argmax(spatial_ir[:, 0])

    # Begin 1 ms before direct sound
    start_index = np.max([0, direct_sample_index + int(np.floor(sample_rate * start_relative_to_direct_ms / 1000))])
    end_index = np.min([spatial_ir.shape[0] - 1, start_index + int(np.floor(sample_rate * duration_ms / 1000))])

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

    plot_angles_rad = np.linspace(-np.pi, np.pi - (2 * np.pi / num_plot_angles), num_plot_angles)
    radii = np.zeros(num_plot_angles)

    # Get energy from the omnidirectional rir channel (this is used for the radius)
    pressure = spatial_ir[start_index:end_index, 0]
    energy_linear = np.square(pressure)

    for angle_index in range(num_plot_angles):
        indices = angles_0toN_wrapped == angle_index
        radii[angle_index] = np.nansum(energy_linear[indices] * np.abs(np.cos(doa_spherical_rad[indices, 2])))

    # window_length = 10
    # radii_wrapped_for_start = radii[-window_length - 1:-1]
    # radii_wrapped_for_end = radii[:window_length]
    # radii_to_smooth = np.concat([radii_wrapped_for_start, radii, radii_wrapped_for_end])
    # radii_smoothed = radii_to_smooth#savgol_filter(radii_to_smooth, 10, 3)
    # radii_smoothed = radii_smoothed[window_length:-window_length]

    # Convert energy radius to decibels, clipping at -60 dB
    radii_dB = 10 * np.log10(np.clip(radii, 1e-6, None))

    return plot_angles_rad, radii_dB