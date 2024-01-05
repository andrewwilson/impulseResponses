import pytest

from audio import load_sample, Sample


@pytest.fixture(scope="module")
def sample() -> Sample:
    sample = load_sample('samples/Norman Mic 0001 [2024-01-01 133335].aif')
    assert sample is not None

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


def test_slice_invalid_num_samples(sample):
    with pytest.raises(AssertionError) as excinfo:
        sample.slice(333.3)
    assert "samples_per_slice must be an integer" in str(excinfo.value)


def test_slice_overlap(sample):
    with pytest.raises(AssertionError) as excinfo:
        sample.slice(100, 1)
    assert "overlap" in str(excinfo.value)


def test_slice_2_parts(sample: Sample):
    samples = sample.samples
    n = samples // 2
    parts = sample.slice(samples_per_slice=n)

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].samples == n
    assert parts[1].samples == n
    assert parts[0].peak <= sample.peak
    assert parts[1].peak <= sample.peak


def test_slice_with_overlap(sample: Sample):
    samples = sample.samples
    n = samples // 2
    parts = sample.slice(samples_per_slice=n, overlap=0.5)

    assert parts is not None
    assert len(parts) == 3
    assert parts[0].samples == n
    assert parts[1].samples == n
    assert parts[2].samples == n
    assert parts[0].peak <= sample.peak
    assert parts[1].peak <= sample.peak
    assert parts[2].peak <= sample.peak
