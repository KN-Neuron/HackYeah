import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class BlinkingClassifier:
    """
    A class to detect blinks in real-time EEG data chunks.
    This version uses a sliding window to apply filters correctly and avoid edge effects.
    """

    def __init__(self, sfreq, channel="Fp1", threshold_uv=100, refractory_samples=125):
        """
        Initializes the blink detector.
        ... (same as before)
        """
        self.sfreq = sfreq
        self.channel = channel
        self.threshold_uv = threshold_uv
        self.channel_names = ["Fp1", "Fp2", "O1", "O2"]
        self.refractory_samples = refractory_samples

    def detect_blink_in_chunk(self, data_window, analysis_duration_samples):
        """
        Analyzes a WINDOW of EEG data to find a blink in the CENTRAL part of it.

        Args:
            data_window (np.ndarray): A 2D numpy array of shape (n_channels, n_samples).
                                      This should be a longer window (e.g., 3 seconds).
            analysis_duration_samples (int): The number of samples in the center of the
                                             window to actually check for blinks (e.g., 250 for 1 sec).

        Returns:
            bool: True if a blink was detected in the central part, False otherwise.
        """
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw_window = mne.io.RawArray(data_window, info, verbose=False)
        raw_window.apply_function(lambda x: x * 1e-6)

        raw_window.filter(l_freq=1.0, h_freq=10.0, fir_design="firwin", verbose=False)

        channel_data_uv = raw_window.get_data(picks=[self.channel])[0] * 1e6

        total_samples = len(channel_data_uv)
        margin = (total_samples - analysis_duration_samples) // 2
        central_data = channel_data_uv[margin : margin + analysis_duration_samples]

        abs_central_data = np.abs(central_data)

        peaks, _ = find_peaks(
            abs_central_data, height=self.threshold_uv, distance=self.refractory_samples
        )

        return len(peaks) > 0
    
    def apply_ICA(self, data_window, analysis_duration_samples):
        """
        Analyzes a WINDOW of EEG data to find a blink in the CENTRAL part of it.

        Args:
            data_window (np.ndarray): A 2D numpy array of shape (n_channels, n_samples).
                                      This should be a longer window (e.g., 3 seconds).
            analysis_duration_samples (int): The number of samples in the center of the
                                             window to actually check for blinks (e.g., 250 for 1 sec).

        Returns:
            bool: True if a blink was detected in the central part, False otherwise.
        """
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw_window = mne.io.RawArray(data_window, info, verbose=False)
        raw_window.apply_function(lambda x: x * 1e-6)

        ica = mne.preprocessing.ICA(n_components=5, max_iter=800)
        ica.fit(raw_window)
        ica.plot_properties(raw)
        
        return

