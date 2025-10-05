import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- IMPORTANT: Install scipy if you haven't already ---
# In your activated venv, run: pip install scipy

class BlinkingClassifier:
    """
    A class to detect blinks in real-time EEG data chunks.
    This version uses a sliding window to apply filters correctly and avoid edge effects.
    """
    def __init__(self, sfreq, channel='Fp1', threshold_uv=100, refractory_samples=125):
        """
        Initializes the blink detector.
        ... (same as before)
        """
        self.sfreq = sfreq
        self.channel = channel
        self.threshold_uv = threshold_uv
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
        ch_names = ['Fp1', 'Fp2', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types='eeg')
        raw_window = mne.io.RawArray(data_window, info, verbose=False)
        raw_window.apply_function(lambda x: x*1e-6)

        # The filter is now applied to a longer signal, which resolves the warning.
        raw_window.filter(l_freq=1., h_freq=10., fir_design='firwin', verbose=False)

        channel_data_uv = raw_window.get_data(picks=[self.channel])[0] * 1e6

        # --- KEY CHANGES FOR ROBUSTNESS ---
        # 1. We only analyze the central part of the window to avoid filter edge artifacts.
        total_samples = len(channel_data_uv)
        margin = (total_samples - analysis_duration_samples) // 2
        central_data = channel_data_uv[margin : margin + analysis_duration_samples]

        # 2. We use the absolute value to detect both positive and negative peaks.
        abs_central_data = np.abs(central_data)

        peaks, _ = find_peaks(
            abs_central_data,
            height=self.threshold_uv,
            distance=self.refractory_samples
        )

        return len(peaks) > 0


def main_simulation():
    """
    Simulates a real-time scenario using an overlapping sliding window.
    """
    # --- 1. Load Data ---
    data_folder = 'eeg_data/test4/'
    try:
        # (Data loading code is the same...)
        file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('_raw.fif')]
        if not file_paths:
            print(f"No .fif files found in '{data_folder}'.")
            return
        raw_list = [mne.io.read_raw_fif(f, preload=True, verbose=False) for f in file_paths]
        raw_full = mne.concatenate_raws(raw_list)
        raw_full.pick_types(eeg=True)
    except FileNotFoundError:
        print(f"The directory '{data_folder}' was not found.")
        return

    sfreq = int(raw_full.info['sfreq'])
    
    # --- !!! TUNING YOUR THRESHOLD (Updated) !!! ---
    # Plot the data converted to µV to see the real peak amplitudes.
    # A negative threshold is not needed as we now check the absolute value.
    
    # raw_full.plot(start=0, duration=20, scalings=dict(eeg=200e-6))
    # plt.show()
    # return
    
    BLINK_THRESHOLD_UV = 10  # <<<--- TUNE THIS VALUE! (Use a positive number)

    # --- 2. Initialize Classifier ---
    classifier = BlinkingClassifier(sfreq=sfreq, threshold_uv=BLINK_THRESHOLD_UV)

    # --- 3. Simulate with Overlapping Sliding Window ---
    window_duration_seconds = 3.0   # How much data the filter sees (e.g., 3s)
    step_duration_seconds = 1.0     # How often we check for a blink (e.g., every 1s)

    window_samples = int(window_duration_seconds * sfreq)
    step_samples = int(step_duration_seconds * sfreq)
    total_samples = raw_full.n_times
    blink_count = 0

    print(f"Analyzing {total_samples / sfreq:.1f} seconds of data...")
    print(f"Using a {window_duration_seconds}s sliding window, checking every {step_duration_seconds}s.")
    print(f"Blink threshold: {BLINK_THRESHOLD_UV} µV.\n")

    for start_sample in range(0, total_samples - window_samples, step_samples):
        end_sample = start_sample + window_samples
        
        data_window = raw_full.get_data(start=start_sample, stop=end_sample)
        
        # We check for blinks in the central `step_samples` part of the window
        if classifier.detect_blink_in_chunk(data_window, analysis_duration_samples=step_samples):
            # The time corresponds to the start of the central analysis block
            current_time = (start_sample + (window_samples - step_samples) // 2) / sfreq
            blink_count += 1
            print(f"--> Blink #{blink_count} detected around {current_time:.1f}s")
            
            # GAME ACTION HERE

    print(f"\nAnalysis complete. Total blinks detected: {blink_count}")


if __name__ == '__main__':
    main_simulation()