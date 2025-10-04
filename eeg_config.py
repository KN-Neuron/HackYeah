"""
Configuration file for BrainAccess Halo 4-channel device.
"""

# BrainAccess Halo 4-channel configuration
BRAINACCESS_HALO_4_CHANNEL = {
    0: "Fp1",  # Left frontal
    1: "Fp2",  # Right frontal
    2: "O1",   # Left occipital
    3: "O2",   # Right occipital
}

# Standard connection port (modify if necessary)
PORT = "/dev/rfcomm0"  # Default for Windows, use appropriate port for your OS
               # On Linux/Mac this might be something like "/dev/ttyUSB0"

# Sampling rate for the device (in Hz)
SAMPLING_RATE = 250

# Path for saving recorded data
DATA_FOLDER_PATH = "eeg_data"

# Active device configuration
USED_DEVICE = BRAINACCESS_HALO_4_CHANNEL
