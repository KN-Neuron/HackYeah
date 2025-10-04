"""
EEG Headset module for BrainAccess Halo 4-channel device.
"""

import os
import time
from typing import Any, Dict, List, Optional

import brainaccess.core as bacore
import numpy as np
from brainaccess.core.eeg_manager import EEGManager
from brainaccess.utils import acquisition

from eeg_config import DATA_FOLDER_PATH, PORT, SAMPLING_RATE, USED_DEVICE


class EEGHeadset:
    """
    Handles connection and data acquisition from BrainAccess Halo 4-channel headset.
    """
    
    def __init__(self, participant_id: str = "test_user") -> None:
        """
        Initialize the EEG headset interface.
        
        Args:
            participant_id (str): ID to use as folder name for saved data.
        """
        self._is_connected: bool = False
        self._is_recording: bool = False
        self._participant_id: str = participant_id
        self._save_dir_path: str = os.path.join(DATA_FOLDER_PATH, participant_id)
        self._connection_attempts: int = 0
        self._max_attempts: int = 3
        self._buffer: List[np.ndarray] = []
        self._annotations: List[Dict[str, Any]] = []
        
        # Initialize BrainAccess library
        print("Initializing BrainAccess library...")
        bacore.init(bacore.Version(2, 0, 0))
        
        # Create directories for data storage
        self._create_dir_if_not_exist(DATA_FOLDER_PATH)
        self._create_dir_if_not_exist(self._save_dir_path)
        
    def connect(self) -> bool:
        """
        Connect to the BrainAccess Halo headset.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if self._is_connected:
            print("Already connected to the headset.")
            return True
            
        print(f"Attempting to connect to BrainAccess Halo on port {PORT}...")
        
        while self._connection_attempts < self._max_attempts:
            try:
                self._eeg_manager = EEGManager()
                self._eeg_acquisition = acquisition.EEG()
                
                # Connect to the headset
                self._eeg_acquisition.setup(self._eeg_manager, USED_DEVICE, port=PORT)
                
                # Check connection
                if self._eeg_manager.is_connected():
                    self._is_connected = True
                    print("Successfully connected to BrainAccess Halo!")
                    return True
                    
            except Exception as e:
                self._connection_attempts += 1
                print(f"Connection attempt {self._connection_attempts} failed: {str(e)}")
                print(f"Retrying in {self._connection_attempts} seconds...")
                time.sleep(self._connection_attempts)
        
        print("Failed to connect to the headset after multiple attempts.")
        print("Please check that:")
        print("1. The device is turned on and charged")
        print("2. The device is within Bluetooth range")
        print("3. The port configuration is correct")
        return False
        
    def disconnect(self) -> None:
        """
        Disconnect from the BrainAccess Halo headset.
        """
        if not self._is_connected:
            print("Not connected to any headset.")
            return
            
        if self._is_recording:
            self.stop_recording()
            
        try:
            self._eeg_manager.disconnect()
            self._is_connected = False
            print("Disconnected from BrainAccess Halo.")
        except Exception as e:
            print(f"Error disconnecting from the headset: {str(e)}")
            
    def start_recording(self, session_name: str = "default_session") -> bool:
        """
        Start recording EEG data.
        
        Args:
            session_name (str): Name of the recording session.
            
        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if not self._is_connected:
            if not self.connect():
                print("Cannot start recording: Failed to connect to the headset.")
                return False
                
        if self._is_recording:
            print("Already recording data.")
            return True
            
        try:
            # Start EEG acquisition
            print("Starting EEG data acquisition...")
            self._eeg_acquisition.start_acquisition()
            self._is_recording = True
            self._session_name = session_name
            self._recording_start_time = time.time()
            
            # Create annotation for session start
            self.annotate_event(f"Session started: {session_name}")
            
            print(f"Recording started for session: {session_name}")
            return True
        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording and save the data.
        
        Returns:
            bool: True if data was saved successfully, False otherwise.
        """
        if not self._is_recording:
            print("No active recording to stop.")
            return False
            
        try:
            # Create annotation for session end
            self.annotate_event("Session ended")
            
            # Get the MNE raw object with recorded data
            print("Processing recorded data...")
            self._eeg_acquisition.get_mne()
            
            # Save data to file
            file_path = os.path.join(
                self._save_dir_path, 
                f"{self._participant_id}_{self._session_name}_{int(self._recording_start_time)}_raw.fif"
            )
            print(f"Saving EEG data to {file_path}")
            self._eeg_acquisition.data.save(file_path)
            
            # Stop acquisition
            self._eeg_acquisition.stop_acquisition()
            self._eeg_manager.clear_annotations()
            self._is_recording = False
            
            print("Recording stopped and data saved successfully.")
            return True
        except Exception as e:
            print(f"Error stopping recording: {str(e)}")
            return False
            
    def annotate_event(self, annotation: str) -> None:
        """
        Add an annotation to the EEG data.
        
        Args:
            annotation (str): Annotation text to add.
        """
        if not self._is_connected:
            print("Cannot annotate: Not connected to the headset.")
            return
            
        try:
            timestamp = time.time() - self._recording_start_time if self._is_recording else 0
            self._eeg_acquisition.annotate(annotation)
            self._annotations.append({
                "timestamp": timestamp,
                "annotation": annotation
            })
            print(f"Annotation added: '{annotation}' at {timestamp:.2f}s")
        except Exception as e:
            print(f"Error adding annotation: {str(e)}")
    
    def get_channel_names(self) -> List[str]:
        """
        Get the names of the EEG channels.
        
        Returns:
            List[str]: List of channel names.
        """
        return list(USED_DEVICE.values())
        
    def get_current_data(self, duration_seconds: float = 1.0) -> np.ndarray:
        """
        Get the most recent EEG data.
        
        Args:
            duration_seconds (float): Amount of data to return in seconds.
            
        Returns:
            np.ndarray: Array of EEG data with shape (channels, samples)
        """
        if not self._is_recording:
            print("Cannot get data: Not currently recording.")
            return np.zeros((len(USED_DEVICE), int(duration_seconds * SAMPLING_RATE)))
            
        try:
            # Use the get_mne method, which can return the last 'tim' seconds of data.
            # We disable annotations here as they are not needed for a quick data sample.
            mne_raw_latest = self._eeg_acquisition.get_mne(tim=duration_seconds, annotations=False)
            
            # The get_mne method returns an MNE Raw object. We need to extract the numpy array.
            if mne_raw_latest:
                data = mne_raw_latest.get_data()
                return data
            else:
                # Handle the case where no data is available yet.
                return np.zeros((len(USED_DEVICE), int(duration_seconds * SAMPLING_RATE)))

        except Exception as e:
            print(f"Error getting current data: {str(e)}")
            return np.zeros((len(USED_DEVICE), int(duration_seconds * SAMPLING_RATE)))
    
    def _create_dir_if_not_exist(self, path: str) -> None:
        """
        Create a directory if it does not exist.
        
        Args:
            path (str): Directory path to create.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
