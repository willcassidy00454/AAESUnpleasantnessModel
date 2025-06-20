import EDC
import Utils


def estimateRT(rir, sample_rate, start_dB = -5, end_dB = -35):
    edc_dB = EDC.getEDC(rir, sample_rate)

    start_index = Utils.findIndexOfClosest(edc_dB, start_dB)
    end_index = Utils.findIndexOfClosest(edc_dB, end_dB)

    sampling_period = 1.0 / sample_rate
    start_time = start_index * sampling_period
    end_time = end_index * sampling_period

    range_dB = end_dB - start_dB
    gradient = range_dB / (end_time - start_time)

    return -60 / gradient