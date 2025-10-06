"""
Demo application for BrainAccess Halo 4-channel EEG headset.
"""

import argparse
import signal
import sys
import time

from eeg_config import PORT
from eeg_headset import EEGHeadset
from eeg_visualizer import EEGVisualizer


def signal_handler(sig, frame):
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
            
    elif args.udp:
        import socket
        import numpy as np
        
        # Create UDP client socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Define the receiver's address
        RECEIVER_ADDRESS = ('127.0.0.1', 11111)
        
        print(f"Starting to stream EEG data via UDP to {RECEIVER_ADDRESS}...")
        
        headset.start_recording(f"demo_session_{int(time.time())}")
        
        try:
            while True:
                try:
                    # Get current data (1 second of data)
                    data_chunk = headset.get_current_data(1)
                    
                    if data_chunk.size > 0:
                        # IMPORTANT FIX: Ensure we always send exactly 4 channels
                        # This handles the case where sometimes get_current_data returns 9 channels
                        if data_chunk.shape[0] > 4:
                            data_chunk = data_chunk[:4, :]  # Take only first 4 channels
                        
                        # Print shape for debugging
                        print(f"Sending data shape: {data_chunk.shape}")
                        
                        # Convert numpy array to bytes before sending
                        message_bytes = data_chunk.astype(np.float64).tobytes()
                        udp_socket.sendto(message_bytes, RECEIVER_ADDRESS)
                        print(f"Sent {len(message_bytes)} bytes of EEG data.")
                except Exception as e:
                    print(f"Error getting or sending data: {e}")
                
                # Wait a short time before next transmission
                time.sleep(0.1)
        finally:
            headset.stop_recording()
            udp_socket.close()
        
    else:
        if not args.no_record:
            print(f"Recording EEG data for {args.duration} seconds...")
            
            # Start recording
            headset.start_recording(f"demo_v2_session_{int(time.time())}")
            
            try:
                for i in range(args.duration):
                    progress = (i + 1) / args.duration
                    bar_length = 30
                    bar = '█' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
                    print(f"\rRecording: [{bar}] {int(progress * 100)}%", end='')
                    
                    if i == 10:
                        headset.annotate_event("10 seconds mark")
                    elif i == 30:
                        headset.annotate_event("30 seconds mark")
                        
                    time.sleep(1)
                    
                print("\nRecording complete.")
            finally:
                headset.stop_recording()
        else:
            print("No action specified. Use --visualize or remove --no-record to perform an action.")
    
    headset.disconnect()
    print("Demo completed.")

if __name__ == "__main__":
    main()
