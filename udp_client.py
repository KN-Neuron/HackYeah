import socket
import numpy as np
import mne

CHANNELS = 4
SFREQ = 250 
MEASUREMENTS_TO_PLOT = 50

DATA_SCALING_FACTOR = 1e-6 

LOW_FREQ = 1.0
HIGH_FREQ = 40.0
SAMPLES_PER_CHANNEL = 250
DTYPE = np.float64
BYTES_PER_NUMBER = np.dtype(DTYPE).itemsize
NUMBERS_PER_ARRAY = CHANNELS * SAMPLES_PER_CHANNEL
REQUIRED_BYTES_FOR_ARRAY = NUMBERS_PER_ARRAY * BYTES_PER_NUMBER

data_buffer = bytearray()
collected_measurements = []
is_running = True

print(f"Collecting {MEASUREMENTS_TO_PLOT} measurements before plotting...")

while is_running:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(1.0)
    message = b'test'
    # NOTE: Using a loopback address for demonstration purposes. 
    # Replace with your device's actual address.
    addr = ("127.0.0.1", 11111) 
    
    # This is a dummy server part for the script to run standalone.
    # In your real use case, you would remove this and just have the client logic.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(addr)
    
    client_socket.sendto(message, addr)
    try:
        dummy_data = np.random.randn(NUMBERS_PER_ARRAY).astype(DTYPE).tobytes()
        server_socket.sendto(dummy_data, addr)

        data, server = client_socket.recvfrom(32768)
        data_buffer.extend(data)
        
        while len(data_buffer) >= REQUIRED_BYTES_FOR_ARRAY:
            chunk_to_process = data_buffer[:REQUIRED_BYTES_FOR_ARRAY]
            numpy_array_1d = np.frombuffer(chunk_to_process, dtype=DTYPE)
            reshaped_array = numpy_array_1d.reshape(CHANNELS, SAMPLES_PER_CHANNEL)
            
            collected_measurements.append(reshaped_array)
            print(f"Successfully processed array of shape {reshaped_array.shape}")
            
            data_buffer = data_buffer[REQUIRED_BYTES_FOR_ARRAY:]

            if len(collected_measurements) >= MEASUREMENTS_TO_PLOT:
                is_running = False
                break
                
    except socket.timeout:
        print('REQUEST TIMED OUT')
    finally:
        client_socket.close()
        server_socket.close()

    if not is_running:
        break

print(f"\nCollected {len(collected_measurements)} measurements. Starting analysis...")

full_data = np.concatenate(collected_measurements, axis=1) * DATA_SCALING_FACTOR
print(f"Final data shape for plotting: {full_data.shape}")

ch_names = [f'CH{i+1}' for i in range(CHANNELS)]
ch_types = ['eeg'] * CHANNELS
info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

raw = mne.io.RawArray(full_data, info)

montage = mne.channels.make_standard_montage('standard_1020')
channel_map = {f'CH{i+1}': name for i, name in enumerate(montage.ch_names[:CHANNELS])}
raw.rename_channels(channel_map)
raw.set_montage(montage)

raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design='firwin')
print(f"Data filtered between {LOW_FREQ} and {HIGH_FREQ} Hz.")

print("\nDisplaying filtered raw signal. Close the plot to continue.")
raw.plot(block=True, title="Filtered Live Stream Data", scalings=dict(eeg=20e-6)) 

print("\nDisplaying Power Spectral Density (PSD). Close the plot to continue.")
raw.compute_psd().plot(spatial_colors=True, average=False, picks='eeg')

print("\nDisplaying sensor locations. Close the plot to continue.")
raw.plot_sensors(ch_type='eeg', show_names=True)


print("\nDisplaying topographical PSD map. Close the plot to continue.")
raw.compute_psd().plot_topo()


print("\nPerforming ICA to find artifacts. This may take a moment...")
ica = mne.preprocessing.ICA(n_components=CHANNELS, random_state=97, max_iter='auto')
ica.fit(raw)

print("Displaying ICA sources. Look for components that resemble blinks or heartbeats.")
ica.plot_sources(raw, block=True)

print("Displaying ICA properties. Close the plot to finish the script.")
ica.plot_properties(raw, picks=[0, 1, 2, 3], block=True)

print("\nScript finished.")
