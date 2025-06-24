from scipy.io import wavfile
import Colouration


def loadRIR(rir_filepath):
    sample_rate, rir = wavfile.read(rir_filepath)
    return sample_rate, rir


def predictUnpleasantness(rir_filepath):
    # Load audio channels and pre-process
    sample_rate, spatial_rir = loadRIR(rir_filepath=rir_filepath)

    # Compute metrics
    colouration = Colouration.getColouration(spatial_rir[:, 0], sample_rate)

    # Compute model
    model_output = colouration

    print(model_output)


if __name__ == "__main__":
    predictUnpleasantness("/Users/willcassidy/Development/GitHub/AAUnpleasantnessModel/Audio/Coloured.wav")