"""
Demo application for BrainAccess Halo 4-channel EEG headset.
"""

import argparse
import signal
import socket
import sys
import time

import numpy as np

from eeg_config import PORT
from eeg_headset import EEGHeadset
from eeg_visualizer import EEGVisualizer


def signal_handler(sig, frame):  # sig and frame params required by signal.signal
    """Handle Ctrl+C to properly close the connection"""
    print("\nShutting down gracefully...")
    if headset._is_recording:
        headset.stop_recording()
    if headset._is_connected:
        headset.disconnect()
    sys.exit(0)

def main():
    print("BrainAccess Halo 4-Channel Demo")
    print("-------------------------------")
    
    parser = argparse.ArgumentParser(description="BrainAccess Halo 4-channel EEG demo")
    parser.add_argument("--port", type=str, default=PORT, help="Serial port for BrainAccess Halo")
    parser.add_argument("--subject", type=str, default="test_subject", help="Subject identifier")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds (default: 60)")
    parser.add_argument("--visualize", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--udp", action="store_true", help="Send recordings on specifired udp client")
    parser.add_argument("--no-record", action="store_true", help="Don't record data to disk")
    
    
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    global headset
    headset = EEGHeadset(participant_id=args.subject)
    
    print(f"Connecting to BrainAccess Halo headset on port {args.port}...")
    if not headset.connect():
        print("Could not connect to the headset. Exiting.")
        return
    
    if args.visualize:
        print("Starting real-time visualization...")
        visualizer = EEGVisualizer(headset)
        
        if not args.no_record:
            print(f"Recording EEG data for subject '{args.subject}'")
            headset.start_recording(f"demo_session_{int(time.time())}")
            
        visualizer.start_visualization()
        
        if headset._is_recording:
            headset.stop_recording()
            
    if args.udp:
        # --- CORRECTED UDP SENDER LOGIC ---
        # This script acts as the client, sending data out.
        
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Define the receiver's (server's) address
        RECEIVER_ADDRESS = ('127.0.0.1', 11111) 
        
        print(f"Starting to stream EEG data via UDP to {RECEIVER_ADDRESS}...")
        
        # Start headset recording to get live data
        headset.start_recording(f"udp_stream_session_{int(time.time())}")
        
        try: 
            while True:
                try:
                    # Get 1 second of data (SFREQ samples)
                    # The argument name is duration_seconds
                    data_chunk = headset.get_current_data(duration_seconds=1.0)
                    
                    if data_chunk.size > 0:
                        # Convert numpy array to bytes before sending
                        message_bytes = data_chunk.astype(np.float64).tobytes()
                        udp_socket.sendto(message_bytes, RECEIVER_ADDRESS)
                        print(f"Sent {len(message_bytes)} bytes of EEG data.")
                        
                except Exception as e:
                    print(f"Error getting or sending data: {e}")

                # Wait for the next interval to send data every 1 second
                time.sleep(1.0)
        finally:
            print("Stopping UDP stream.")
            headset.stop_recording()
            udp_socket.close()
        
    else:
        if not args.no_record:
            print(f"Recording EEG data for {args.duration} seconds...")
            
            # Start recording
            headset.start_recording(f"demo_v2_session_{int(time.time())}")
            
            try:
                # Display a simple progress bar
                for i in range(args.duration):
                    progress = (i + 1) / args.duration
                    bar_length = 30
                    bar = '█' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
                    print(f"\rRecording: [{bar}] {int(progress * 100)}%", end='')
                    
                    # Add annotations at certain points
                    if i == 10:
                        headset.annotate_event("10 seconds mark")
                    elif i == 30:
                        headset.annotate_event("30 seconds mark")
                        
                    time.sleep(1)
                    
                print("\nRecording complete.")
            finally:
                # Stop recording and save data
                headset.stop_recording()
        else:
            print("No action specified. Use --visualize or remove --no-record to perform an action.")
    
    # Disconnect from headset
    headset.disconnect()
    print("Demo completed.")

if __name__ == "__main__":
    main()
