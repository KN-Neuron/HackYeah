import socket
import numpy as np
import mne

# --- Configuration ---
CHANNELS = 4
SFREQ = 250
SAMPLES_PER_SECOND = 250

# --- Network Configuration ---
DTYPE = np.float64
BYTES_PER_NUMBER = np.dtype(DTYPE).itemsize  # 8 bytes for float64
BUFFER_SIZE = 20000  # Large enough for incoming packets

# --- SERVER ADDRESS ---
SERVER_ADDRESS = ("127.0.0.1", 11111)

# --- Create socket ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind(SERVER_ADDRESS)

print(f"UDP client listening on {SERVER_ADDRESS}")
collected_measurements = []

try:
    while len(collected_measurements) < 10:  # Collect 10 measurements
        # Wait for a packet
        data, address = client_socket.recvfrom(BUFFER_SIZE)
        print(f"Received {len(data)} bytes from {address}")
        
        try:
            # Convert bytes back to numpy array
            numpy_array = np.frombuffer(data, dtype=DTYPE)
            
            # Calculate actual samples received (may be more than expected)
            samples_received = len(numpy_array) // CHANNELS
            
            # Reshape to proper dimensions
            reshaped_array = numpy_array.reshape(CHANNELS, samples_received)
            
            collected_measurements.append(reshaped_array)
            print(f"Successfully processed array of shape {reshaped_array.shape}")
            
        except Exception as e:
            print(f"Error processing data: {e}")
            
finally:
    client_socket.close()
    
# Process collected measurements with MNE if needed
if collected_measurements:
    # Concatenate all data chunks
    full_data = np.concatenate(collected_measurements, axis=1)
    print(f"Final data shape: {full_data.shape}")
    
    # Create MNE info and continue processing...
    ch_names = ['Fp1', 'Fp2', 'O1', 'O2']
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(full_data, info)
    
    # Plot data
    raw.plot(block=True, scalings=dict(eeg=20e-6))