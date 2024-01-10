import librosa
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


def plot_frequency_response(signal: np.array, sr: int, label=None, logscale=False):
    # Compute the frequency response
    # FFT and frequency bins
    freq_response = fft(signal)
    freq_bins = np.fft.fftfreq(len(signal), 1 / sr)

    # Plot the frequency response
    # Only plot the positive frequencies and convert to dB
    positive_freqs = freq_bins > 0
    magnitude = 20 * np.log10(np.abs(freq_response[positive_freqs]))

    plotfn = plt.semilogx if logscale else plt.plot
    plotfn(freq_bins[positive_freqs], magnitude, label=label, alpha=0.3)

    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")


def plot_sample_spectrogram(signal: Sample):
    return plot_spectrogram(signal.signal, signal.sr, signal.name)


def plot_spectrogram(signal: np.array, sr: int, title: str = None):
    D = librosa.stft(signal, )  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
