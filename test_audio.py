import os

import numpy as np
import pytest

import audio
from audio import load_sample, Sample, filter_slices
import tempfile


@pytest.fixture(scope="module")
def sample() -> Sample:
    sample = load_sample('samples/Eastman piezo 0009 [2023-12-30 162118].aif')
    assert sample is not None
    assert sample.duration == 16.0
    return sample


@pytest.fixture(scope="module")
def sample2() -> Sample:
    sample = load_sample('samples/Eastman mic 0009 [2023-12-30 162118].aif')
    assert sample is not None
    assert sample.duration == 16.0

    return sample


def test_str(sample: Sample):
    s = str(sample)
    assert s is not None
    assert sample.name in s


def test_info(sample: Sample):
    info = sample.info()
    assert info is not None
    assert info['name'] == sample.name
    assert info['peak'] == sample.peak
    assert info['rms'] == sample.rms
    assert info['loudness'] == sample.loudness


def test_slices_invalid_num_samples(sample):
    with pytest.raises(AssertionError) as excinfo:
        sample._slices(333.3)
    assert "samples_per_slice must be an integer" in str(excinfo.value)


def test_slices_overlap(sample):
    with pytest.raises(AssertionError) as excinfo:
        sample._slices(100, 1)
    assert "overlap" in str(excinfo.value)


def test_slices_2_parts(sample: Sample):
    samples = sample.samples
    n = samples // 2
    parts = sample._slices(samples_per_slice=n)

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].samples == n
    assert parts[1].samples == n
    assert parts[0].peak <= sample.peak
    assert parts[1].peak <= sample.peak


def test_slices_by_duration_2_parts(sample: Sample):
    duration = sample.duration
    parts = sample.slices(duration / 2, overlap=0.0)
    assert len(parts) == 2
    assert parts[0].duration == duration / 2
    assert parts[1].duration == duration / 2


def test_slices_with_overlap(sample: Sample):
    samples = sample.samples
    n = samples // 2
    parts = sample._slices(samples_per_slice=n, overlap=0.5)

    assert parts is not None
    assert len(parts) == 3
    assert parts[0].samples == n
    assert parts[1].samples == n
    assert parts[2].samples == n
    assert parts[0].peak <= sample.peak
    assert parts[1].peak <= sample.peak
    assert parts[2].peak <= sample.peak


def test_loudness_for_short_slice(sample: Sample):
    samples = sample._slices(samples_per_slice=audio.duration_to_samples(0.01, sample.sr))
    l = samples[0].loudness
    assert np.isnan(l)


def test_filter_slices(sample: Sample):
    slices = sample.slices(0.1, 0.0)
    assert len(slices) == 160
    filtered = filter_slices(slices, 1, measure_fn=lambda s: s.peak)
    assert len(filtered) == 1

    filtered = filter_slices(slices, 0, measure_fn=lambda s: s.peak)
    assert len(filtered) == 160


def test_prepare_slices_no_overlap_no_discard(sample: Sample, sample2: Sample):
    # samples are 16 sec

    # slice with no overlap and no thresholding
    slices = audio.prepare_slices(sample, sample2, 1.0, 0.0,
                                  -24, 0, True,
                                  lambda n: np.kaiser(n, 14))
    assert (len(slices) == 16)  # no overlap and no threshold filter
    assert len(slices[0]) == 2


def test_prepare_slices_with_overlap_and_no_discard(sample: Sample, sample2: Sample):
    # samples are 16 sec
    slices = audio.prepare_slices(sample, sample2, 1.0, 0.9,
                                  -24.0, 0, True,
                                  lambda n: np.kaiser(n, 14))
    assert len(list(slices)) == 151


def test_prepare_slices_with_overlap_and_discard(sample: Sample, sample2: Sample):
    # samples are 16 sec
    slices = audio.prepare_slices(sample, sample2, 1.0, 0.9,
                                  -24.0, 0.3, True,
                                  lambda n: np.kaiser(n, 14))
    assert len(list(slices)) == 144


def test_create_average_ir(sample: Sample, sample2: Sample):
    slices = audio.prepare_slices(sample, sample2, 1.0, 0,
                                  -24.0, 0.3, True,
                                  np.blackman)
    ir = audio.create_average_ir(slices)
    assert len(ir) == 2048


def test_create_ir(sample: Sample, sample2: Sample):
    ir = audio.create_ir(sample, sample2)
    assert len(ir) == 2048


def test_distance_between(sample: Sample, sample2: Sample):
    assert audio.distance_between(sample, sample) == 0
    assert audio.distance_between(sample2, sample2) == 0

    dist = audio.distance_between(sample, sample2)
    assert dist - 0.0018364589 < 1e-8

    source = sample
    target = sample2.normalise_loudness(source.loudness)
    dist = audio.distance_between(source, target)
    assert dist == 0.15075707


def test_apply_ir(sample: Sample, sample2: Sample):
    dist = audio.distance_between(sample, sample2)
    ir = audio.create_ir(sample, sample2)
    processed = sample.apply_ir(ir)
    processed_dist = audio.distance_between(sample2, processed)
    assert (processed_dist, dist) == (0, 0)


def test_save_as_is_lossless(sample: Sample):
    file = tempfile.mktemp(".wav", "tmpfoo")
    sample.save_as(file)
    loaded_sample = audio.load_sample(file)
    assert np.abs(sample.signal - loaded_sample.signal).sum() == 0
    os.unlink(file)
