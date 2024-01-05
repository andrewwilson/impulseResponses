import matplotlib.pyplot as plt
import numpy as np

from audio import Sample


def plot_signal(sample: Sample, title=None, label=None, start=0, length=None):
    """
     Function to plot an audio signal
    """
    title = title if title is not None else sample.name
    length = sample.samples if length is None else length
    sig = sample.signal[start:start + length]
    plt.figure(figsize=(20, 4))
    plt.plot(np.linspace(0, len(sig) / sample.sr, num=len(sig)), sig, label=label)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
