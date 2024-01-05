import os
from typing import List, Any

import pandas as pd
import librosa
import numpy as np
import pyloudnorm


def duration_to_samples(duration: float, sr: float) -> int:
    return int(sr * duration)


class Sample():
    def __init__(self, signal: np.array, sr: float, name: str = None):
        self.signal = signal
        self.sr = sr
        self.name = name

    def __str__(self):
        return f"[{self.name} {self.sr}khz {self.duration:.2f}s]"

    def __repr__(self):
        return self.__str__()

    @property
    def samples(self):
        return len(self.signal)

    @property
    def duration(self):
        '''
        duration in seconds
        :return: the sample duration in seconds
        '''
        return self.samples / self.sr

    @property
    def loudness(self):
        try:
            meter = pyloudnorm.Meter(self.sr)
            return meter.integrated_loudness(self.signal)
        except ValueError as ex:
            # loudness not available for short samples..
            if "Audio must have length greater than the block size" in str(ex):
                return np.nan
            raise ex

    @property
    def rms(self):
        return np.sqrt(np.mean(self.signal ** 2))

    @property
    def peak(self):
        return np.max(np.abs(self.signal))

    def info(self) -> pd.Series:
        ser = pd.Series()
        ser['name'] = self.name
        ser['samples'] = self.samples
        ser['duration'] = self.duration
        ser['peak'] = self.peak
        ser['rms'] = self.rms
        ser['loudness'] = self.loudness

        return ser

    def slices(self, samples_per_slice: int, overlap: float = 0.0) -> List[Any]:
        """
        creates a list of slices of this sample.
        :param samples_per_slice:
        :param overlap:
        :return:
        """
        assert 0 <= overlap < 1, f"overlap must be >= 0 and < 1"
        assert type(samples_per_slice) == int, f"samples_per_slice must be an integer: {samples_per_slice}"
        assert samples_per_slice > 3

        step_size = int(samples_per_slice * (1 - overlap))

        parts = []
        for i in range(0, len(self.signal) - samples_per_slice + 1, step_size):
            signal_part = self.signal[i:i + samples_per_slice]
            parts.append(signal_part)
        return [Sample(signal_part, self.sr, self.name + f" {i + 1:3d}/{len(parts)}")
                for i, signal_part in enumerate(parts)]


def apply_window(signal_section, window):
    return signal_section * window


def load_sample(filename: str, sr=None) -> Sample:
    """
    :param filename:
    :param sr: sample rate to normalise to. pass None to load at original sample rate
    :return: signal, sr
    """
    # pass sr=None to ensure we keep the recorded sample rate
    signal, sr = librosa.load(filename, sr=sr)
    return Sample(signal, sr, filename)


def info_grid(samples: List[Sample]) -> pd.DataFrame:
    """
    Returns a dataframe with a row of info for each samples
    :param samples:
    :return: pandas DataFrame of sample info rows.
    """
    return pd.DataFrame([s.info() for s in samples])


if __name__ == '__main__':

    sample_dir = 'samples'
    samples = []
    for filename in [f for f in os.listdir('samples') if f.endswith('.aif') or f.endswith('.wav')]:
        sample = load_sample(os.path.join(sample_dir, filename))
        samples.append(sample)

    print(info_grid(samples))
