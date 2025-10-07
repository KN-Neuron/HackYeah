import socket
import warnings
from scipy.signal import find_peaks
import time
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque

CHANNELS = 4
SFREQ = 250
DATA_SCALING_FACTOR = 1e-6
LOW_FREQ = 1.0
HIGH_FREQ = 40.0

SAMPLES_PER_CHANNEL = 250
DTYPE = np.float64
BYTES_PER_NUMBER = np.dtype(DTYPE).itemsize
NUMBERS_PER_ARRAY = CHANNELS * SAMPLES_PER_CHANNEL
REQUIRED_BYTES_FOR_ARRAY = NUMBERS_PER_ARRAY * BYTES_PER_NUMBER

PLOT_WINDOW_SECONDS = 5
SAMPLES_TO_PLOT = PLOT_WINDOW_SECONDS * SFREQ
collected_measurements = deque(maxlen=int(SAMPLES_TO_PLOT / SAMPLES_PER_CHANNEL) + 2)

UDP_IP = "192.168.43.22"
UDP_PORT = 11111
addr = (UDP_IP, UDP_PORT)

ch_names = [f'CH{i+1}' for i in range(CHANNELS)]
ch_types = ['eeg'] * CHANNELS
info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
mne.set_log_level('warning')


montage = mne.channels.make_standard_montage('standard_1020')
mapped_ch_names = montage.ch_names[:CHANNELS]
channel_map = {f'CH{i+1}': mapped_ch_names[i] for i in range(CHANNELS)}


plt.ion()
fig, axes = plt.subplots(CHANNELS, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Live EEG Data Feed', fontsize=16)
is_paused = False

class ButtonHandler:
    def __init__(self):
        self.is_paused = False

    def toggle_pause(self, event):
        self.is_paused = not self.is_paused
        print(f"Plotting {'Paused' if self.is_paused else 'Resumed'}")

button_handler = ButtonHandler()

data_buffer = bytearray()
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.settimeout(1.0)

print("Starting real-time EEG data plotting...")
print("Press the 'Pause/Resume' button on the plot to freeze the feed.")
print("Press Ctrl+C in the terminal to stop the script.")

ax_button = fig.add_axes([0.8, 0.01, 0.15, 0.05])
btn_pause = Button(ax_button, 'Pause/Resume')
btn_pause.on_clicked(button_handler.toggle_pause)


FRONTAL_CHANNEL_INDEX = 0
BLINK_THRESHOLD = 30e-6  

try:
    previous_segment = None
    while True:

        time.sleep(1)
        if not button_handler.is_paused:
            client_socket.sendto(b"message", addr)
            try:
                data, server = client_socket.recvfrom(8192)
                data_buffer.extend(data)

                while len(data_buffer) >= REQUIRED_BYTES_FOR_ARRAY:
                    chunk = data_buffer[:REQUIRED_BYTES_FOR_ARRAY]
                    data_buffer = data_buffer[REQUIRED_BYTES_FOR_ARRAY:]

                    reshaped_array = np.frombuffer(chunk, dtype=DTYPE).reshape(CHANNELS, SAMPLES_PER_CHANNEL)
                    collected_measurements.append(reshaped_array)

            except socket.timeout:
                print('REQUEST TIMED OUT')
                continue


            if collected_measurements:
                full_data = np.concatenate(list(collected_measurements), axis=1) * DATA_SCALING_FACTOR

                if full_data.shape[1] > SAMPLES_TO_PLOT:
                    full_data = full_data[:, -SAMPLES_TO_PLOT:]



                raw = mne.io.RawArray(full_data, info, verbose=False)
                raw.rename_channels(channel_map)
                raw.set_montage(montage, on_missing='warn')
                raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design='firwin', verbose=False)

                frontal_data = raw.get_data(picks=[FRONTAL_CHANNEL_INDEX])[0]

                if np.max(np.abs(frontal_data[-SAMPLES_PER_CHANNEL:])) > BLINK_THRESHOLD:
                    print("\n\n\n\n\n blink \n\n\n\n\n")


                plot_data, times = raw.get_data(return_times=True)

                for i, ax in enumerate(axes):
                    ax.clear()
                    ax.plot(times, plot_data[i])
                    ax.set_ylabel(f"{raw.ch_names[i]}\n(uV)")

                    min_val, max_val = np.min(plot_data[i]), np.max(plot_data[i])
                    padding = (max_val - min_val) * 0.1
                    ax.set_ylim(min_val - padding, max_val + padding)
                    ax.grid(True)

                axes[-1].set_xlabel("Time (s)")
                fig.tight_layout(rect=[0, 0.05, 1, 0.96])


        plt.pause(0.1)


except KeyboardInterrupt:
    print("\nScript interrupted by user. Closing socket and exiting.")
finally:
    client_socket.close()
    plt.ioff()
    plt.show()
    print("Socket closed. Script finished.")