import mne
import numpy as np


class StressClassifier:
    """
    Calculates a stress percentage based on the Beta/Alpha power ratio
    from frontal EEG channels.
    """

    BETA_BAND = (13.0, 30.0)
    ALPHA_BAND = (8.0, 13.0)

    def __init__(
        self, sfreq, channels=["Fp1", "Fp2"], calm_threshold=0.5, stress_threshold=1.5
    ):
        """
        Initializes the stress detector.

        Args:
            sfreq (int): The sampling frequency of the EEG data.
            channels (list): List of frontal channels to use for analysis.
            calm_threshold (float): The Beta/Alpha ratio considered very calm.
            stress_threshold (float): The Beta/Alpha ratio considered very stressed.
        """
        self.sfreq = sfreq
        self.channels = channels
        self.calm_threshold = calm_threshold
        self.stress_threshold = stress_threshold

    def get_stress_percentage(self, data_chunk):
        """
        Analyzes a chunk of EEG data and returns a stress percentage.

        Args:
            data_chunk (np.ndarray): 2D numpy array (channels, samples) of EEG data.

        Returns:
            tuple: (percentage, ratio)
        """
        ch_names = ["Fp1", "Fp2", "O1", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw_chunk = mne.io.RawArray(data_chunk, info, verbose=False)
        raw_chunk.pick_channels(self.channels)

        # Filter to the relevant frequency range
        raw_chunk.filter(
            l_freq=self.ALPHA_BAND[0],
            h_freq=self.BETA_BAND[1],
            fir_design="firwin",
            verbose=False,
        )

        psd, freqs = mne.time_frequency.psd_welch(
            raw_chunk, fmin=2.0, fmax=40.0, n_fft=self.sfreq, verbose=False
        )

        alpha_power = np.mean(
            psd[:, (freqs >= self.ALPHA_BAND[0]) & (freqs < self.ALPHA_BAND[1])]
        )
        beta_power = np.mean(
            psd[:, (freqs >= self.BETA_BAND[0]) & (freqs < self.BETA_BAND[1])]
        )

        if alpha_power < 1e-12:
            return (0.0, 0.0)

        ratio = beta_power / alpha_power

        # Normalize the ratio to a 0-100% scale
        if ratio <= self.calm_threshold:
            percentage = 0.0
        elif ratio >= self.stress_threshold:
            percentage = 100.0
        else:
            percentage = (
                (ratio - self.calm_threshold)
                / (self.stress_threshold - self.calm_threshold)
            ) * 100

        return (percentage, ratio)
