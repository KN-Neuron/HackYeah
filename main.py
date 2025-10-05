import asyncio
import json
import numpy as np
import collections
import websockets

from BlinkingClassifier import BlinkingClassifier
from TirednessRegression import TirednessClassifier
from Stress import StressClassifier
from Attention import AttentionClassifier

# --- Configuration ---
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
UDP_HOST = "127.0.0.1"
UDP_PORT = 11111

SAMPLING_RATE = 250
CHANNELS = 4

# Analysis windows
BUFFER_DURATION_SEC = 5.0
ANALYSIS_INTERVAL_SEC = 1.0
TIREDNESS_WINDOW_SEC = 5.0
STRESS_WINDOW_SEC = 5.0
BLINK_WINDOW_SEC = 3.0

# --- Global variables ---
clients = set()
results_queue = asyncio.Queue()  # Use asyncio's queue

# --- WebSocket Server Logic ---
async def register_client(websocket, path):
    clients.add(websocket)
    print(f"[WebSocket] Client connected: {websocket.remote_address}. Total: {len(clients)}")
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print(f"[WebSocket] Client disconnected. Total: {len(clients)}")

async def broadcast_data(data):
    if clients:
        message = json.dumps(data)
        # Use asyncio.gather for concurrent sending
        await asyncio.gather(*[client.send(message) for client in clients])

async def websocket_broadcaster():
    """Waits for data from the queue and sends it to clients."""
    while True:
        data = await results_queue.get()
        print(f"[WebSocket] Broadcasting: Blink={data['is_blinking']}, Tiredness={data['tiredness']['percentage']}%")
        await broadcast_data(data)

# --- UDP Server and Analysis Logic ---
class EEGUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
        # Initialize classifiers
        self.blinker = BlinkingClassifier(sfreq=SAMPLING_RATE, threshold_uv=75)
        self.tiredness_analyzer = TirednessClassifier(sfreq=SAMPLING_RATE, channels=['O1', 'O2'])
        self.stress_analyzer = StressClassifier(sfreq=SAMPLING_RATE, channels=['Fp1', 'Fp2'])
        self.attention_analyzer = AttentionClassifier(sfreq=SAMPLING_RATE)

        # Data buffer
        self.buffer_size = int(BUFFER_DURATION_SEC * SAMPLING_RATE)
        self.data_buffer = collections.deque(maxlen=self.buffer_size)
        
        # Window sizes in samples
        self.analysis_samples = int(ANALYSIS_INTERVAL_SEC * SAMPLING_RATE)
        self.tiredness_samples = int(TIREDNESS_WINDOW_SEC * SAMPLING_RATE)
        self.stress_samples = int(STRESS_WINDOW_SEC * SAMPLING_RATE)
        self.blink_window_samples = int(BLINK_WINDOW_SEC * SAMPLING_RATE)

    def datagram_received(self, data, addr):
        """This is called by the event loop when a UDP packet arrives."""
        print(f"[UDP] Received {len(data)} bytes from {addr}")
        
        # Decode the byte data into a numpy array (1D)
        numpy_1d = np.frombuffer(data, dtype=np.float64)
        
        # Expected number of float64 values for 1 second of data
        expected_size = CHANNELS * self.analysis_samples
        
        if numpy_1d.size == expected_size:
            reshaped = numpy_1d.reshape(CHANNELS, self.analysis_samples)
            
            # Add new samples to the rolling buffer
            for i in range(reshaped.shape[1]):
                self.data_buffer.append(reshaped[:, i])
            
            # Run analysis if the buffer is full
            if len(self.data_buffer) >= self.buffer_size:
                self.run_analysis()
        else:
            print(f"[UDP] Warning: Received packet of incorrect size. Got {numpy_1d.size}, expected {expected_size}")

    def run_analysis(self):
        """Runs all classifiers on the current data buffer."""
        np_buffer = np.array(self.data_buffer).T

        # --- Blink detection ---
        blink_data_window = np_buffer[:, -self.blink_window_samples:]
        is_blinking = self.blinker.detect_blink_in_chunk(blink_data_window, self.analysis_samples)

        # --- Tiredness analysis ---
        tiredness_data_chunk = np_buffer[:, -self.tiredness_samples:]
        tiredness_percentage, tiredness_ratio = self.tiredness_analyzer.get_tiredness_percentage(tiredness_data_chunk)

        # --- Stress analysis ---
        stress_data_chunk = np_buffer[:, -self.stress_samples:]
        stress_percentage, stress_ratio = self.stress_analyzer.get_stress_percentage(stress_data_chunk)
            
        # --- Attention analysis ---
        attention_percentage, _ = self.attention_analyzer.get_attention_percentage(np_buffer)
        
        # --- Prepare and queue data for broadcasting ---
        output_data = {
            "timestamp": time.time(),
            "is_blinking": is_blinking,
            "tiredness": {"percentage": round(tiredness_percentage, 2), "ratio": round(tired_ratio, 3)},
            "stress": {"percentage": round(stress_percentage, 2), "ratio": round(stress_ratio, 3)},
            "attention": {"percentage": round(attention_percentage, 2)}
        }
        
        # Put result in the async queue
        self.queue.put_nowait(output_data)

# --- Main Program Entry Point ---
async def main_async():
    """Main async function to run UDP and WebSocket servers."""
    loop = asyncio.get_running_loop()

    # Start the UDP server endpoint
    udp_transport, _ = await loop.create_datagram_endpoint(
        lambda: EEGUDPProtocol(results_queue),
        local_addr=(UDP_HOST, UDP_PORT)
    )
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT}")

    # Start the WebSocket server
    ws_server = await websockets.serve(register_client, WEBSOCKET_HOST, WEBSOCKET_PORT)
    print(f"WebSocket server started on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    # Run the broadcaster task
    try:
        await websocket_broadcaster()
    finally:
        udp_transport.close()
        ws_server.close()
        await ws_server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        print("Program shut down.")