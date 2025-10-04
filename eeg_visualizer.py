import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from scipy import signal


def plot_raw_eeg(raw_data: mne.io.Raw, duration: float = 10.0, start: float = 0.0) -> Figure:
    """
    Plot a segment of raw EEG data.
    
    Args:
        raw_data: MNE Raw object containing EEG data
        duration: Duration in seconds to plot
        start: Start time in seconds
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Plot data
    raw_data.plot(duration=duration, start=start, scalings='auto', show=False, block=False)
    
    return fig

def plot_spectrogram(raw_data: mne.io.Raw, fmin: float = 0, fmax: float = 50) -> Figure:
    """
    Plot a spectrogram of the EEG data.
    
    Args:
        raw_data: MNE Raw object containing EEG data
        fmin: Minimum frequency to display
        fmax: Maximum frequency to display
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create spectrograms for each channel
    picks = mne.pick_types(raw_data.info, eeg=True)
    channel_names = [raw_data.ch_names[i] for i in picks]
    
    for i, ch_idx in enumerate(picks):
        plt.subplot(len(picks), 1, i + 1)
        data, times = raw_data[ch_idx, :]
        
        # Calculate spectrogram
        fs = raw_data.info['sfreq']
        nperseg = min(int(fs * 2), data.shape[1])  # 2-second segments
        
        f, t, Sxx = signal.spectrogram(data.flatten(), fs=fs, nperseg=nperseg)
        
        # Plot spectrogram
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.ylim(fmin, fmax)
        plt.ylabel(f'{channel_names[i]}\nFrequency (Hz)')
        plt.colorbar(label='Power (dB)')
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    
    return fig

def plot_power_spectrum(raw_data: mne.io.Raw, fmin: float = 0, fmax: float = 50) -> Figure:
    """
    Plot the power spectrum of the EEG data.
    
    Args:
        raw_data: MNE Raw object containing EEG data
        fmin: Minimum frequency to display
        fmax: Maximum frequency to display
        
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Calculate power spectrum for each channel
    psds, freqs = mne.time_frequency.psd_welch(raw_data, fmin=fmin, fmax=fmax, 
                                              n_fft=2048, n_overlap=1024)
    
    # Convert to dB
    psds = 10 * np.log10(psds)
    
    # Plot each channel
    for i in range(psds.shape[0]):
        plt.plot(freqs, psds[i, :], label=raw_data.ch_names[i])
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('EEG Power Spectrum')
    plt.legend()
    plt.grid(True)
    
    # Mark frequency bands
    band_dict = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    
    y_min, y_max = plt.ylim()
    for band, (f_low, f_high) in band_dict.items():
        if f_high >= fmin and f_low <= fmax:
            plt.fill_betweenx([y_min, y_max], f_low, f_high, color='gray', alpha=0.2)
            plt.text((f_low + f_high) / 2, y_max - 2, band, 
                     horizontalalignment='center')
    
    plt.tight_layout()
    return fig

def plot_brain_map(raw_data: mne.io.Raw, freq_band: Tuple[float, float] = (8, 13)) -> Optional[Figure]:
    """
    Plot a topographic map of the brain for a specific frequency band.
    
    Args:
        raw_data: MNE Raw object containing EEG data
        freq_band: Tuple with (low, high) frequency range to plot
        
    Returns:
        fig: Matplotlib figure or None if too few channels
    """
    # Check if we have enough channels for a meaningful topographic map
    # For BrainAccess Halo with 4 channels, this is not ideal but we'll do our best
    if len(raw_data.ch_names) < 4:
        print("Not enough channels for brain mapping")
        return None
    
    try:
        # Calculate power in the frequency band
        psds, freqs = mne.time_frequency.psd_welch(raw_data, fmin=freq_band[0], 
                                                  fmax=freq_band[1], n_fft=2048)
        
        # Average power across the specified frequency band
        psds_mean = np.mean(psds, axis=1)
        
        # Create montage for plotting
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage)
        
        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Plot topographic map
        mne.viz.plot_topomap(psds_mean, raw_data.info, cmap='viridis', 
                            show=False, contours=0)
        
        plt.title(f'Brain Activity: {freq_band[0]}-{freq_band[1]} Hz')
        plt.colorbar(label='Power (µV²/Hz)')
        
        return fig
    except Exception as e:
        print(f"Error creating brain map: {str(e)}")
        return None

def analyze_frequency_bands(raw_data: mne.io.Raw) -> Dict[str, Dict[str, float]]:
    """
    Analyze the power in different frequency bands for each channel.
    
    Args:
        raw_data: MNE Raw object containing EEG data
        
    Returns:
        dict: Dictionary with channel names and band powers
    """
    # Define frequency bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    
    # Calculate PSD
    psds, freqs = mne.time_frequency.psd_welch(raw_data, fmin=0.5, fmax=50, 
                                              n_fft=2048, n_overlap=1024)
    
    # For each channel, calculate the average power in each frequency band
    result = {}
    for i, ch_name in enumerate(raw_data.ch_names):
        ch_result = {}
        for band_name, (fmin, fmax) in bands.items():
            # Find frequencies in the band
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            # Calculate mean power in the band
            band_power = np.mean(psds[i, idx_band])
            ch_result[band_name] = band_power
        result[ch_name] = ch_result
        
    return result
