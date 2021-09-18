import tqdm
import scipy.signal
from utils import *
import biosppy
import os

def wave_to_spec(wave, fs, nperseg):
    f, t, Sxx = scipy.signal.spectrogram(wave, fs=fs, nperseg=nperseg)
    return Sxx


def lead_to_spec(filter, fs, nperseg, lead):
    result = []
    for i in range(12):
        spec = lead[:, i]
        if filter:
            spec = _filter(lead[:, i])
        spec = wave_to_spec(spec, fs, nperseg)
        result.append(spec)
    result = np.stack(result)
    return result / result.max()


def _filter(lead):
    try:
        _, filtered, _, _, _, _, _ = biosppy.signals.ecg.ecg(
            lead,
            sampling_rate=100,
            show=False
        )
        return filtered
    except:
        pass
    return lead



def transform(params):
    filter = params["filter"]
    fs = params["fs"]
    nperseg = params["nperseg"]

    train, test, _ = read_data('../data')

    train_wave = read_wave("../data/ecg/" + train["Id"] + ".npy")
    test_wave = read_wave("../data/ecg/" + test["Id"] + ".npy")

    info("contain dir: ./data/{}/".format(params["out_dir_name"]))

    # if directory does not exist, create it
    if not os.path.exists("./data/{}/".format(params["out_dir_name"])):
        os.makedirs("./data/{}/".format(params["out_dir_name"]))

    print('transform train data...')
    for i in tqdm.tqdm(range(train.shape[0])):
        np.save('./data/{}/train/wave-{}.npy'.format(params["out_dir_name"], i), lead_to_spec(filter, fs, nperseg, train_wave[i]))

    print('transform test data...')
    for i in tqdm.tqdm(range(test.shape[0])):
        np.save('./data/{}/test/wave-{}.npy'.format(params["out_dir_name"], i), lead_to_spec(filter, fs, nperseg, test_wave[i]))




