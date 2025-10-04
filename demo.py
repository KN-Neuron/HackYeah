import os
import time

import matplotlib.pyplot as plt
import numpy as np

from eeg_config import DEFAULT_PORT
from eeg_headset import EEGHeadset
from eeg_visualizer import (analyze_frequency_bands, plot_power_spectrum,
                            plot_raw_eeg, plot_spectrogram)


def main():
    print("BrainAccess Halo 4-Channel Demo")
    print("-------------------------------")
    
    # Create participant ID based on timestamp
    participant_id = f"participant_{int(time.time())}"
    
    # Initialize headset
    headset = EEGHeadset(participant_id)
    
    # Connect to headset
    if not headset.connect(port=DEFAULT_PORT):
        print("Failed to connect to headset. Exiting.")
        return
    
    # Show connection status
    headset.show_connection_status()
    
    # Start recording
    if not headset.start_recording(block_name="demo_recording"):
        print("Failed to start recording. Exiting.")
        headset.disconnect()
        return
    
    # Start real-time visualization
    print("Starting real-time EEG visualization...")
    headset.visualize_real_time()
    
    # Record for a short time with annotations
    print("\nRecording EEG data. Please sit still for 30 seconds.")
    print("The demo will add annotations at specific times.")
    
    try:
        # Record with annotations to demonstrate different states
        time.sleep(5)
        headset.annotate_event("Eyes Open")
        print("Annotation: Eyes Open")
        
        time.sleep(5)
        headset.annotate_event("Eyes Closed")
        print("Annotation: Eyes Closed (Alpha waves should increase)")
        
        time.sleep(10)
        headset.annotate_event("Mental Calculation")
        print("Annotation: Mental Calculation (please perform 23 × 17 in your head)")
        
        time.sleep(5)
        headset.annotate_event("Relaxation")
        print("Annotation: Relaxation (please relax and breathe deeply)")
        
        time.sleep(5)
        
        # Display frequency spectrum during recording
        print("\nShowing current frequency spectrum...")
        headset.show_frequency_spectrum()
        
        # Continue recording
        print("\nContinuing recording for a few more seconds...")
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    
    # Stop recording and save data
    print("\nStopping recording...")
    file_path = headset.stop_recording()
    
    if not file_path:
        print("Failed to save recording. Exiting.")
        headset.disconnect()
        return
    
    print(f"Recording saved to: {file_path}")
    
    # Show connection status after recording
    headset.show_connection_status()
    
    # Analyze the recorded data
    print("\nAnalyzing recorded data...")
    try:
        # Load the saved data
        import mne
        raw_data = mne.io.read_raw_fif(file_path, preload=True)
        
        # Plot raw EEG
        print("Plotting raw EEG data...")
        fig_raw = plot_raw_eeg(raw_data, duration=30, start=0)
        fig_raw.suptitle('Raw EEG Data')
        plt.figure(fig_raw.number)
        plt.savefig(os.path.join(os.path.dirname(file_path), 'raw_eeg_plot.png'))
        plt.show()
        
        # Plot spectrogram
        print("Generating spectrogram...")
        fig_spec = plot_spectrogram(raw_data)
        fig_spec.suptitle('EEG Spectrogram')
        plt.figure(fig_spec.number)
        plt.savefig(os.path.join(os.path.dirname(file_path), 'spectrogram.png'))
        plt.show()
        
        # Plot power spectrum
        print("Generating power spectrum...")
        fig_psd = plot_power_spectrum(raw_data)
        plt.figure(fig_psd.number)
        plt.savefig(os.path.join(os.path.dirname(file_path), 'power_spectrum.png'))
        plt.show()
        
        # Analyze frequency bands
        print("\nFrequency band analysis:")
        band_powers = analyze_frequency_bands(raw_data)
        
        # Print band power analysis
        for channel, bands in band_powers.items():
            print(f"\nChannel: {channel}")
            for band, power in bands.items():
                print(f"  {band}: {power:.2f} µV²/Hz")
                
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
    
    # Disconnect from headset
    print("\nDisconnecting from headset...")
    headset.disconnect()
    print("Demo completed.")

if __name__ == "__main__":
    main()
