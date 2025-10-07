import os
import mne
import numpy as np


# --- The TirednessClassifier class, now without artifact rejection ---
class TirednessClassifier:
    """
    Calculates a tiredness percentage by normalizing the (Alpha + Theta) / Beta ratio.
    WARNING: This version has NO artifact rejection and will produce unreliable
             results on noisy data.
    """

    ALPHA_BAND = (8.0, 13.0)
    THETA_BAND = (4.0, 8.0)
    BETA_BAND = (13.0, 30.0)

    def __init__(
        self, sfreq, channels=["O1", "O2"], alert_threshold=1.0, tired_threshold=3.0
    ):
        """
        Initializes the tiredness detector with pre-defined thresholds.
        """
        self.sfreq = sfreq
        self.channels = channels
        self.alert_threshold = alert_threshold
        self.tired_threshold = tired_threshold

    def get_tiredness_percentage(self, data_chunk):
        """
        Analyzes a chunk of EEG data and returns a tiredness percentage.
        This version will analyze every chunk, even if it's noisy.
        """
        ch_names = ["Fp1", "Fp2", "O1", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw_chunk = mne.io.RawArray(data_chunk, info, verbose=False)
        raw_chunk.pick_channels(self.channels)

        psd, freqs = mne.time_frequency.psd_welch(
            raw_chunk, fmin=2.0, fmax=40.0, n_fft=self.sfreq, verbose=False
        )

        alpha_power = np.mean(
            psd[:, (freqs >= self.ALPHA_BAND[0]) & (freqs < self.ALPHA_BAND[1])]
        )
        theta_power = np.mean(
            psd[:, (freqs >= self.THETA_BAND[0]) & (freqs < self.THETA_BAND[1])]
        )
        beta_power = np.mean(
            psd[:, (freqs >= self.BETA_BAND[0]) & (freqs < self.BETA_BAND[1])]
        )

        if beta_power < 1e-12:
            return (0.0, 0.0)

        ratio = (alpha_power + theta_power) / beta_power

        # Normalize the ratio to a 0-100% scale
        if ratio <= self.alert_threshold:
            percentage = 0.0
        elif ratio >= self.tired_threshold:
            percentage = 100.0
        else:
            percentage = (
                (ratio - self.alert_threshold)
                / (self.tired_threshold - self.alert_threshold)
            ) * 100

        return (percentage, ratio)


def estimate_tiredness_ignoring_noise():
    """
    Loads EEG files and provides an overall tiredness percentage without
    rejecting any data for artifacts.
    """
    print("--- Starting Tiredness Estimation (Noise Rejection DISABLED) ---")

    TARGET_DATA_FOLDER = "eeg_data/test4/"

    # Using general population-based thresholds as a guess
    ALERT_THRESHOLD = 1.0
    TIRED_THRESHOLD = 3.0

    try:
        file_paths = [
            os.path.join(TARGET_DATA_FOLDER, f)
            for f in os.listdir(TARGET_DATA_FOLDER)
            if f.endswith("_raw.fif")
        ]
        if not file_paths:
            print(f"ERROR: No .fif files found in '{TARGET_DATA_FOLDER}'.")
            return
        raw_list = [
            mne.io.read_raw_fif(f, preload=True, verbose=False) for f in file_paths
        ]
        raw_full = mne.concatenate_raws(raw_list)
        raw_full.pick_types(eeg=True)
    except FileNotFoundError:
        print(f"The directory '{TARGET_DATA_FOLDER}' was not found.")
        return

    sfreq = int(raw_full.info["sfreq"])

    classifier = TirednessClassifier(
        sfreq=sfreq,
        channels=["O1", "O2"],
        alert_threshold=ALERT_THRESHOLD,
        tired_threshold=TIRED_THRESHOLD,
    )

    window_samples = int(5.0 * sfreq)
    step_samples = int(2.0 * sfreq)
    total_samples = raw_full.n_times

    all_percentages = []
    print(f"Analyzing {len(file_paths)} file(s) from '{TARGET_DATA_FOLDER}'...")
    print(
        "WARNING: Artifact rejection is disabled. Results will be affected by noise.\n"
    )

    for start in range(0, total_samples - window_samples, step_samples):
        end = start + window_samples
        data_chunk = raw_full.get_data(start=start, stop=end)

        percentage, ratio = classifier.get_tiredness_percentage(data_chunk)
        all_percentages.append(percentage)

        # Optional: uncomment to see the percentage for each chunk
        # current_time = end / sfreq
        # print(f"Time: {current_time:5.1f}s | Raw Ratio: {ratio:4.2f} | Tiredness: {percentage:3.0f}%")

    # --- Display the Final Result ---
    if not all_percentages:
        print("--- Analysis Failed ---")
        print(
            "Reason: Could not calculate any ratios. The data may be completely flat or corrupted."
        )
    else:
        overall_tiredness = np.mean(all_percentages)

        print("--- Analysis Complete ---")

        bar = "█" * int(overall_tiredness / 5) + "░" * (20 - int(overall_tiredness / 5))
        print(f"\nOverall Tiredness Estimate: {overall_tiredness:.0f}%")
        print(f"[{bar}]")

        print(
            "\n**CRUCIAL CAVEAT:** This number is likely NOT a measure of brain-based tiredness."
        )
        print(
            "Because noise rejection was disabled, it is primarily a measure of how much low-frequency"
        )
        print(
            "noise (from movement, muscle tension, or bad electrode contact) is in the recording."
        )


if __name__ == "__main__":
    estimate_tiredness_ignoring_noise()
