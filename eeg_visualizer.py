"""
Real-time visualization of BrainAccess Halo EEG data.
"""

import threading
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eeg_config import SAMPLING_RATE
from eeg_headset import EEGHeadset


class EEGVisualizer:
    """
    Real-time visualization for BrainAccess Halo 4-channel EEG data.
    """
    
    def __init__(self, headset: EEGHeadset, window_size: float = 5.0):
        """
        Initialize the EEG visualizer.
        
        Args:
            headset (EEGHeadset): EEG headset object to get data from.
            window_size (float): Time window to display in seconds.
        """
        self.headset = headset
        self.window_size = window_size
        self.channels = headset.get_channel_names()
        self.num_channels = len(self.channels)
        self.sample_count = int(window_size * SAMPLING_RATE)
        self.time_vector = np.linspace(-window_size, 0, self.sample_count)
        
        # Data buffer for each channel
        self.data_buffer = np.zeros((self.num_channels, self.sample_count))
        
        # For frequency analysis
        self.fft_size = 256
        self.freq_vector = np.fft.rfftfreq(self.fft_size, 1.0 / SAMPLING_RATE)
        self.freq_data = np.zeros((self.num_channels, len(self.freq_vector)))
        
        # For signal quality
        # self.signal_quality = [100] * self.num_channels
        
        self.is_running = False
        self.animation = None
        self.fig = None
    
    def start_visualization(self):
        """
        Start real-time visualization of EEG data.
        """
        if self.is_running:
            print("Visualization is already running.")
            return
            
        # Connect to headset if not already connected
        if not self.headset._is_connected:
            if not self.headset.connect():
                print("Cannot start visualization: Failed to connect to headset.")
                return
        
        # Set up the figure and subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle('BrainAccess Halo 4-Channel EEG Visualization', fontsize=16)
        
        # Grid layout for plots
        gs = self.fig.add_gridspec(3, 4)
        
        # Time-domain signals (top row)
        self.time_axes = []
        for i in range(self.num_channels):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(f"Channel {self.channels[i]}")
            ax.set_ylim(-100, 100)  # μV range
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (μV)")
            ax.grid(True)
            line, = ax.plot(self.time_vector, self.data_buffer[i], 'b-')
            self.time_axes.append((ax, line))
        
        # Frequency domain (middle row)
        self.freq_axes = []
        for i in range(self.num_channels):
            ax = self.fig.add_subplot(gs[1, i])
            ax.set_title(f"{self.channels[i]} - Frequency Spectrum")
            ax.set_xlim(0, 50)  # Hz range
            ax.set_ylim(0, 20)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.grid(True)
            line, = ax.plot(self.freq_vector, self.freq_data[i], 'r-')
            self.freq_axes.append((ax, line))
        
        # Brain activity heatmap (bottom left)
        self.brain_ax = self.fig.add_subplot(gs[2, :2])
        self.brain_ax.set_title("Brain Activity Heatmap")
        self.brain_ax.axis('off')
        
        # Create a simple head outline
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, edgecolor='black')
        self.brain_ax.add_patch(circle)
        
        # Add channel positions
        positions = {
            "Fp1": (0.35, 0.8), "Fp2": (0.65, 0.8),
            "O1": (0.35, 0.2),  "O2": (0.65, 0.2)
        }
        
        self.brain_dots = {}
        for ch_name, pos in positions.items():
            dot = plt.Circle(pos, 0.05, fill=True, color='blue', alpha=0.7)
            self.brain_ax.add_patch(dot)
            self.brain_dots[ch_name] = dot
            self.brain_ax.text(pos[0], pos[1] + 0.07, ch_name, ha='center')
        
        # Signal quality indicator (bottom right)
        self.quality_ax = self.fig.add_subplot(gs[2, 2:])
        self.quality_ax.set_title("Signal Quality")
        self.quality_ax.set_xlim(0, 1)
        self.quality_ax.set_ylim(0, self.num_channels)
        self.quality_ax.set_yticks(np.arange(self.num_channels) + 0.5)
        self.quality_ax.set_yticklabels(self.channels)
        self.quality_ax.set_xlabel("Quality (%)")
        
        # self.quality_bars = self.quality_ax.barh(
        #     np.arange(self.num_channels) + 0.5,
        #     self.signal_quality,
        #     height=0.8,
        #     color=['green'] * self.num_channels
        # )
        
        # Text for connection status
        self.status_text = self.fig.text(
            0.5, 0.02,
            "Status: Connected",
            ha='center',
            bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 5}
        )
        
        # Start animation
        self.is_running = True
        self.animation = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=100,  # Update every 100ms
            blit=False
        )
        
        # Start recording if not already
        if not self.headset._is_recording:
            self.headset.start_recording("visualization_session")
        
        plt.tight_layout()
        plt.show(block=True)
    
    def stop_visualization(self):
        """
        Stop the visualization and clean up resources.
        """
        if not self.is_running:
            return
            
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
            
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            
        self.is_running = False
        print("Visualization stopped.")
    
    def _update_plot(self, frame):
        """
        Update function for the animation.
        
        Args:
            frame: Frame number (not used)
        """
        try:
            # Get latest EEG data
            new_data = self.headset.get_current_data(0.1)  # Get 100ms of data
            
            if new_data.size == 0:
                return
                
            # Roll the buffer and add new data
            samples_to_add = new_data.shape[1]
            self.data_buffer = np.roll(self.data_buffer, -samples_to_add, axis=1)
            self.data_buffer[:, -samples_to_add:] = new_data
            
            # Update time domain plots
            for i, (ax, line) in enumerate(self.time_axes):
                line.set_ydata(self.data_buffer[i])
            
            # Calculate FFT for frequency domain
            for i in range(self.num_channels):
                # Use last FFT_SIZE samples for frequency analysis
                channel_data = self.data_buffer[i, -self.fft_size:]
                # Apply hamming window
                windowed_data = channel_data * np.hamming(len(channel_data))
                # Calculate FFT
                fft_result = np.abs(np.fft.rfft(windowed_data)) / self.fft_size
                # Square to get power
                self.freq_data[i] = fft_result**2
                
                # Update frequency domain plot
                self.freq_axes[i][1].set_ydata(self.freq_data[i])
            
            # Update brain activity heatmap
            alpha_values = []
            for i, ch_name in enumerate(self.channels):
                # Calculate alpha activity (8-12 Hz) power
                alpha_idx = np.logical_and(self.freq_vector >= 8, self.freq_vector <= 12)
                alpha_power = np.mean(self.freq_data[i][alpha_idx])
                alpha_values.append(alpha_power)
                
            # Normalize alpha values to color scale
            if max(alpha_values) > 0:
                norm_alpha = [val / max(alpha_values) for val in alpha_values]
            else:
                norm_alpha = [0] * len(alpha_values)
                
            # Update dot colors based on alpha activity
            for i, ch_name in enumerate(self.channels):
                # Map power to color: blue (low) to red (high)
                r = min(1.0, norm_alpha[i] * 2)
                b = max(0.0, 1.0 - norm_alpha[i] * 2)
                self.brain_dots[ch_name].set_color((r, 0, b))
                
            # Update signal quality indicators
            # In a real implementation, get actual signal quality
            # Here we simulate quality changes
            for i in range(self.num_channels):
                # Simulate quality between 60-100%
                quality = 60 + np.random.rand() * 40
                # self.signal_quality[i] = quality
                
                # Update bar length and color
                self.quality_bars[i].set_width(quality / 100)
                
                # Set color based on quality
                if quality > 80:
                    color = 'green'
                elif quality > 60:
                    color = 'yellow'
                else:
                    color = 'red'
                self.quality_bars[i].set_color(color)
                
            # Update status text
            if self.headset._is_connected:
                self.status_text.set_text(f"Status: Connected - Recording Time: {int(time.time() - self.headset._recording_start_time)}s")
                self.status_text.set_bbox(dict(facecolor='green', alpha=0.5, pad=5))
            else:
                self.status_text.set_text("Status: Disconnected")
                self.status_text.set_bbox(dict(facecolor='red', alpha=0.5, pad=5))
            
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            
        return []
