import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

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


def plot_ir_frequency_response(ir:np.array, sr:int, label=None, logscale=False):
    # Compute the frequency response
    # FFT and frequency bins
    freq_response = fft(ir)
    freq_bins = np.fft.fftfreq(len(ir), 1 / sr)

    # Plot the frequency response
    # Only plot the positive frequencies and convert to dB
    positive_freqs = freq_bins > 0
    magnitude = 20 * np.log10(np.abs(freq_response[positive_freqs]))

    plotfn = plt.semilogx if logscale else plt.plot
    plotfn(freq_bins[positive_freqs], magnitude, label=label)

    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

