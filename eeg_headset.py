import os
import threading
import time
from typing import Any, Dict, List, Optional

import brainaccess.core as bacore
import matplotlib.pyplot as plt
import numpy as np
from brainaccess.core.eeg_manager import EEGManager
from brainaccess.utils import acquisition
from matplotlib.animation import FuncAnimation
from scipy import signal

from eeg_config import DATA_FOLDER_PATH, DEFAULT_PORT, USED_DEVICE


class EEGHeadset:
    def __init__(self, participant_id: str) -> None:
        """
        Initialize the BrainAccess Halo headset.
        
        Args:
            participant_id (str): ID to use as folder name for saved data.
        """
        self._is_started = False
        self._is_connected = False
        self._is_recording = False
        self._buffer_size = 1000  # 5 seconds of data at 200 Hz
        
        # Create buffers for visualization
        self._data_buffer = {ch_name: np.zeros(self._buffer_size) for ch_idx, ch_name in USED_DEVICE.items()}
        self._viz_active = False

        self._save_dir_path = os.path.join(DATA_FOLDER_PATH, participant_id)
        
        # Initialize library
        bacore.init(bacore.Version(2, 0, 0))
        self._create_dir_if_not_exist(DATA_FOLDER_PATH)
        self._create_dir_if_not_exist(self._save_dir_path)
        
        # For visualization
        self.fig = None
        self.axes = None
        self.lines = None

    def _create_dir_if_not_exist(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def connect(self, port: str = DEFAULT_PORT) -> bool:
        """
        Connect to the BrainAccess Halo headset.
        
        Args:
            port (str): Serial port to connect to.
            
        Returns:
            bool: True if connection successful, False otherwise.
        """
        print("Connecting to BrainAccess Halo headset...")
        try:
            self._eeg_manager = EEGManager()
            self._eeg_acquisition = acquisition.EEG()
            
            # Connect to the headset
            self._eeg_acquisition.setup(self._eeg_manager, USED_DEVICE, port=port)
            self._is_connected = True
            print("Successfully connected to BrainAccess Halo headset")
            
            # Get device info
            device_info = self._eeg_manager.get_device_info()
            print(f"Device info: {device_info}")
            
            # Check impedance
            self.check_impedance()
            
            return True
        except Exception as e:
            print(f"Failed to connect: {str(e)}")
            return False

    def check_impedance(self) -> Dict[str, float]:
        """
        Check electrode impedance.
        
        Returns:
            Dict[str, float]: Dictionary with channel names and impedance values.
        """
        if not self._is_connected:
            print("Cannot check impedance: headset not connected")
            return {}
            
        print("Checking electrode impedance...")
        impedance_values = {}
        
        try:
            # This is a placeholder - actual implementation would depend on BrainAccess API
            for ch_idx, ch_name in USED_DEVICE.items():
                # Simulate impedance reading (replace with actual API call)
                impedance = self._eeg_manager.get_impedance(ch_idx) if hasattr(self._eeg_manager, 'get_impedance') else np.random.randint(5, 50)
                impedance_values[ch_name] = impedance
                quality = "Good" if impedance < 10 else "Fair" if impedance < 30 else "Poor"
                print(f"Channel {ch_name}: {impedance} kΩ - {quality}")
                
            return impedance_values
        except Exception as e:
            print(f"Error checking impedance: {str(e)}")
            return {}

    def start_recording(self, block_name: str = "default") -> bool:
        """
        Start recording EEG data.
        
        Args:
            block_name (str): Name for the recording block.
            
        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if not self._is_connected:
            print("Cannot start recording: headset not connected")
            return False
            
        if self._is_recording:
            print("Already recording")
            return False
            
        try:
            print(f"Starting EEG recording: {block_name}")
            self._block_name = block_name
            self._eeg_acquisition.start_acquisition()
            self._is_recording = True
            self._start_time = time.time()
            
            # Start data collection thread for visualization
            if not self._viz_active:
                self._data_collection_thread = threading.Thread(target=self._collect_data)
                self._viz_active = True
                self._data_collection_thread.daemon = True
                self._data_collection_thread.start()
                
            return True
        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            return False

    def _collect_data(self):
        """Background thread to collect data for visualization."""
        while self._viz_active and self._is_connected:
            try:
                if self._is_recording:
                    # Get the latest data from the device
                    # This is a placeholder - the actual implementation would depend on the BrainAccess API
                    latest_data = self._eeg_acquisition.get_latest_samples(10)
                    if latest_data:
                        for ch_idx, ch_name in USED_DEVICE.items():
                            # Update the buffer with the new data
                            self._data_buffer[ch_name] = np.roll(self._data_buffer[ch_name], -len(latest_data[ch_idx]))
                            self._data_buffer[ch_name][-len(latest_data[ch_idx]):] = latest_data[ch_idx]
            except Exception as e:
                print(f"Error collecting data: {str(e)}")
            time.sleep(0.05)  # Update every 50ms

    def stop_recording(self) -> str:
        """
        Stop recording and save the data.
        
        Returns:
            str: Path to the saved file or empty string if error.
        """
        if not self._is_recording:
            print("Not recording")
            return ""
            
        try:
            print("Stopping EEG recording and saving data...")
            
            # Get all recorded data as MNE raw object
            raw_data = self._eeg_acquisition.get_mne()
            
            # Save data to file
            file_path = os.path.join(
                self._save_dir_path, f"EEG_{self._block_name}_{int(self._start_time)}_raw.fif"
            )
            raw_data.save(file_path, overwrite=True)
            
            # Stop acquisition
            self._eeg_acquisition.stop_acquisition()
            self._is_recording = False
            print(f"Data saved to {file_path}")
            
            return file_path
        except Exception as e:
            print(f"Error stopping recording: {str(e)}")
            return ""

    def disconnect(self) -> bool:
        """
        Disconnect from the headset.
        
        Returns:
            bool: True if disconnected successfully, False otherwise.
        """
        if not self._is_connected:
            print("Headset not connected")
            return True
            
        try:
            if self._is_recording:
                self.stop_recording()
                
            # Stop visualization
            self._viz_active = False
            if hasattr(self, '_data_collection_thread') and self._data_collection_thread.is_alive():
                self._data_collection_thread.join(timeout=1.0)
                
            print("Disconnecting from headset...")
            self._eeg_manager.disconnect()
            self._is_connected = False
            print("Disconnected successfully")
            return True
        except Exception as e:
            print(f"Error disconnecting: {str(e)}")
            return False

    def annotate_event(self, event_description: str) -> bool:
        """
        Add an annotation/marker to the EEG data.
        
        Args:
            event_description (str): Description of the event.
            
        Returns:
            bool: True if annotation was added successfully, False otherwise.
        """
        if not self._is_recording:
            print("Cannot annotate: not recording")
            return False
            
        try:
            print(f"Adding annotation: {event_description}")
            self._eeg_acquisition.annotate(event_description)
            return True
        except Exception as e:
            print(f"Error adding annotation: {str(e)}")
            return False

    def visualize_real_time(self) -> None:
        """Start real-time visualization of EEG data."""
        if not self._is_connected:
            print("Cannot visualize: headset not connected")
            return
            
        # Create figure for visualization
        self.fig, self.axes = plt.subplots(len(USED_DEVICE), 1, figsize=(10, 8), sharex=True)
        if len(USED_DEVICE) == 1:
            self.axes = [self.axes]  # Make iterable for single channel case
            
        # Create line for each channel
        self.lines = {}
        time_vector = np.linspace(0, self._buffer_size / 200, self._buffer_size)  # Assuming 200Hz sampling rate
        
        for i, (ch_idx, ch_name) in enumerate(USED_DEVICE.items()):
            line, = self.axes[i].plot(time_vector, self._data_buffer[ch_name])
            self.axes[i].set_ylabel(f"{ch_name} (μV)")
            self.axes[i].set_ylim(-100, 100)
            self.lines[ch_name] = line
            
        self.axes[-1].set_xlabel("Time (s)")
        self.fig.suptitle("BrainAccess Halo Real-time EEG")
        
        # Function to update the plot
        def update_plot(frame):
            for ch_name, line in self.lines.items():
                line.set_ydata(self._data_buffer[ch_name])
            return tuple(self.lines.values())
            
        # Create animation
        self.ani = FuncAnimation(self.fig, update_plot, interval=100, blit=True)
        plt.tight_layout()
        plt.show(block=False)

    def show_frequency_spectrum(self) -> None:
        """Show frequency spectrum of current EEG data."""
        if not self._is_connected or not self._is_recording:
            print("Cannot show spectrum: not connected or not recording")
            return
            
        try:
            # Create figure for frequency spectrum
            plt.figure(figsize=(10, 6))
            
            for ch_idx, ch_name in USED_DEVICE.items():
                # Get data from buffer
                eeg_data = self._data_buffer[ch_name]
                
                # Calculate power spectrum
                fs = 200  # Sampling frequency (Hz)
                f, Pxx = signal.welch(eeg_data, fs, nperseg=min(256, len(eeg_data)))
                
                # Plot only 0-50 Hz range
                mask = f <= 50
                plt.semilogy(f[mask], Pxx[mask], label=ch_name)
                
            plt.title('EEG Power Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (μV²/Hz)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error showing frequency spectrum: {str(e)}")

    def show_connection_status(self) -> None:
        """Display a visual representation of the connection status."""
        if not self._is_connected:
            status_color = 'red'
            status_text = 'Disconnected'
        elif not self._is_recording:
            status_color = 'orange'
            status_text = 'Connected, Not Recording'
        else:
            status_color = 'green'
            status_text = 'Connected, Recording'
            
        plt.figure(figsize=(5, 2))
        plt.axis('off')
        plt.text(0.5, 0.5, f"Status: {status_text}", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16,
                 color=status_color,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
        plt.tight_layout()
        plt.show()
