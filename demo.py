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
            
    elif args.udp:
        import socket
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('10.181.159.22', 11111))
        
        headset.start_recording(f"demo_session_{int(time.time())}")
        try: 
            while True:
                _, address = server_socket.recvfrom(1024)
                try:
                    message = headset.get_current_data(duration_second=1)
                    server_socket.sendto(message, address)
                    print(message[0:10])
                except AttributeError as e:
                    print(f"Error getting current data: {e}")
                headset.annotate_event("1 seconds mark")
                # time.sleep(1)
        finally:
            headset.stop_recording()
        
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
