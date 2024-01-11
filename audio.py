import os
from typing import List, Any, Tuple

import pandas as pd
import librosa
import numpy as np
import pyloudnorm
import soundfile
from scipy.signal import convolve
from scipy.fft import fft, ifft
import logging

import audio

logger = logging.getLogger(__name__)

_MAX_PEAK = 10.0


def duration_to_samples(duration: float, sr: int) -> int:
    return int(sr * duration)


class Sample():
    def __init__(self, signal: np.array, sr: int, name: str = None):
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

    def normalise_loudness(self, target_loudness: float):
        new_sig = pyloudnorm.normalize.loudness(self.signal, self.loudness, target_loudness)
        return Sample(new_sig, self.sr, self.name + "(loudness normed)")

    def with_signal(self, new_signal, name=None):
        """ create a sample with the given signal
        """
        name = name if name is not None else 'processed ' + self.name
        return Sample(new_signal, self.sr, name)

    @property
    def rms(self):
        return np.sqrt(np.mean(self.signal ** 2))

    @property
    def peak(self):
        return np.max(np.abs(self.signal))

    def info(self) -> pd.Series:
        ser = pd.Series()
        ser['name'] = self.name
        ser['sr'] = self.sr
        ser['samples'] = self.samples
        ser['duration'] = self.duration
        ser['peak'] = self.peak
        ser['rms'] = self.rms
        ser['loudness'] = self.loudness

        return ser

    def slice_by_duration(self, start_duration: float, end_duration: float, name: str = None) -> Any:
        start_idx = audio.duration_to_samples(start_duration, self.sr)
        end_idx = audio.duration_to_samples(end_duration, self.sr)
        new_sig = self.signal[start_idx:end_idx]
        new_name = name if name is not None else self.name + f"[{start_idx}:{end_idx}]"
        return self.with_signal(new_sig, new_name)

    def slices(self, duration: float, overlap: float = 0.0) -> List[Any]:
        """
        creates a list of slices by duration and overlap
        :param duration:
        :param overlap:
        :return:
        """
        samples = duration_to_samples(duration, self.sr)
        return self._slices(samples_per_slice=samples, overlap=overlap)

    def split_at(self, at_time: float) -> Tuple[Any, Any]:
        assert at_time < self.duration, 'split point time should be less than sample duration'
        split_index = duration_to_samples(at_time, self.sr)
        sig1 = self.signal[0:split_index]
        sig2 = self.signal[split_index:]
        return self.with_signal(sig1), self.with_signal(sig2)

    def save_as(self, filename, sample_rate=None):
        sample_rate = sample_rate if sample_rate is not None else self.sr
        soundfile.write(filename, self.signal, sample_rate, subtype='PCM_24')

    def _slices(self, samples_per_slice: int, overlap: float = 0.0) -> List[Any]:
        """
        creates a list of slices of this sample, by number of slices and overlap
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

    def apply_ir(self, impulse_response):
        processed_signal = convolve(self.signal, impulse_response)
        # processed signal will naturally be len(signal)+len(impulse_response)-1 long
        assert len(processed_signal) == len(self.signal) + len(impulse_response) - 1
        # truncate it to preserve signal length
        processed_signal = processed_signal[:len(self.signal)]
        # check we got that right
        assert len(processed_signal) == len(self.signal)
        return self.with_signal(processed_signal)


def filter_slices(slices: List[Sample], threshold: float, measure_fn) -> List[Sample]:
    assert 0 <= threshold <= 1, 'threshold must be between 0 and 1'
    measured_samples = [(measure_fn(slice), slice) for slice in slices]
    max_val = max([m for (m, s) in measured_samples])
    limit_val = threshold * max_val
    return [s for (m, s) in measured_samples if m >= limit_val]


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


def prepare_slices(source: Sample, target: Sample,
                   slice_duration: float,
                   slice_overlap: float,
                   normalise_loudness_to: float | None,
                   amplitude_threshold: float,
                   normalise_slices: bool,
                   window_fn) -> List[Tuple[Sample, Sample]]:
    """
    Prepares a selection of slices from a source and target sample
    :param source:
    :param target:
    :param slice_duration:
    :param slice_overlap:
    :param amplitude_threshold: only keep samples with amplitude greater than
    or equal to the max of the whole sample. This is to discard patches of relative silence.
    :param normalise_loudness_to: the loudness level to which to normalise both samples
    :param normalise_slices: should the individual slices also be normalised
    :param window_fn: the window function to apply to the samples or None
    :return: list of tuples of (source_slice, target_slice)
    """
    assert 0 <= amplitude_threshold <= 1, 'threshold must be between 0 and 1'
    assert source.duration == target.duration, f"source and target don't have same duration: {source} != {target}"

    if normalise_loudness_to is not None:
        source = source.normalise_loudness(normalise_loudness_to)
        target = target.normalise_loudness(normalise_loudness_to)

        # check we haven't made peak signal too loud
        assert source.peak <= _MAX_PEAK, f'source peak signal > {_MAX_PEAK}: {source.peak}'
        assert target.peak <= _MAX_PEAK, f'target peak signal > {_MAX_PEAK}: {target.peak}'

    source_slices = source.slices(slice_duration, slice_overlap)
    target_slices = target.slices(slice_duration, slice_overlap)

    # discard samples that are too quiet.
    target_slice_info = info_grid(target_slices)
    # use the target indices into the target signal, and apply to both source and
    # target slices as we must keep them the same
    keep_idx = [slice.rms >= amplitude_threshold * target_slice_info.rms.max()
                for slice in target_slices]

    source_slices = [s for i, s in enumerate(source_slices) if keep_idx[i]]
    target_slices = [s for i, s in enumerate(target_slices) if keep_idx[i]]

    # optionally normalise the individual slice loudness
    if normalise_slices:
        source_slices = [s.normalise_loudness(normalise_loudness_to) for s in source_slices]
        target_slices = [s.normalise_loudness(normalise_loudness_to) for s in target_slices]

    all_slices = source_slices + target_slices
    # assert that all slices are same length
    slice_lengths = [s.samples for s in all_slices]
    min_slice_length = min(slice_lengths)
    max_slice_length = max(slice_lengths)

    # assert that slice peaks are still <= 1.0
    for s in all_slices:
        assert s.peak <= _MAX_PEAK, f"slice has peak > {_MAX_PEAK}: {s.peak}"

    assert min_slice_length == max_slice_length, \
        f'slices were not same length: min: {min_slice_length} max: {max_slice_length}'
    # apply window function
    if window_fn is not None:
        window = window_fn(max_slice_length)
        source_slices = [s.with_signal(window * s.signal) for s in source_slices]
        target_slices = [s.with_signal(window * s.signal) for s in target_slices]

    return list(zip(source_slices, target_slices))


def create_ir(source: Sample, target: Sample, max_len=2048) -> np.array:
    assert source.samples == target.samples
    assert source.sr == target.sr

    fft_dry = fft(source.signal)
    fft_target = fft(target.signal)
    epsilon = 1e-10
    impulse_response_fft = fft_target / (fft_dry + epsilon)
    impulse_response = np.real(ifft(impulse_response_fft))
    return impulse_response[:max_len]


def create_average_ir(slices: List[Tuple[Sample, Sample]]) -> np.array:
    irs = [audio.create_ir(source, target) for source, target in slices]
    avg_ir = np.average(irs, axis=0)
    return avg_ir


def distance_between(sample1: Sample, sample2: Sample) -> float:
    assert sample1.samples == sample2.samples, "number of samples differ"
    assert sample1.sr == sample2.sr, "sample rates differ"
    difference = sample1.signal - sample2.signal
    metric = np.sqrt(np.mean(difference ** 2))
    return metric


if __name__ == '__main__':

    sample_dir = 'samples'
    samples = []
    for filename in [f for f in os.listdir('samples') if f.endswith('.aif') or f.endswith('.wav')]:
        sample = load_sample(os.path.join(sample_dir, filename))
        samples.append(sample)

    print(info_grid(samples))
