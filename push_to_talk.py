#!/usr/bin/env python3
import os
import sys
import warnings
import subprocess
import threading
import time
import queue
import select

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to devnull during imports to suppress ALSA/CUDA messages
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    import evdev
    from evdev import InputDevice, categorize, ecodes
    import pyaudio
    import numpy as np
    from faster_whisper import WhisperModel
finally:
    # Restore stderr
    sys.stderr.close()
    sys.stderr = stderr

def send_to_tmux(text):
    """Send text to the currently focused tmux pane"""
    try:
        # Get currently focused pane
        result = subprocess.run(
            ['tmux', 'display-message', '-p', '#{pane_id}'],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            pane_id = result.stdout.strip()
            # Send text to the focused pane
            subprocess.run(
                ['tmux', 'send-keys', '-t', pane_id, text],
                timeout=1
            )
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Tmux not available or command failed
        pass
    return False

def send_to_xdo(text):
    """Send text to the currently focused window using xdotool"""
    try:
        # Type the text to the currently focused window
        subprocess.run(
            ['xdotool', 'type', '--', text],
            timeout=1
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # xdotool not available or command failed
        pass
    return False

def find_keyboard_devices():
    """Find all keyboard devices"""
    devices = []
    for path in evdev.list_devices():
        try:
            device = InputDevice(path)
            capabilities = device.capabilities()
            # Check if device has KEY events
            if ecodes.EV_KEY in capabilities:
                # Check if it has alphabetic keys (indicates it's a keyboard)
                if ecodes.KEY_A in capabilities[ecodes.EV_KEY]:
                    devices.append(device)
        except (OSError, PermissionError):
            continue
    return devices

class PushToTalkRecorder:
    def __init__(self, hotkey='alt', model='medium.en', device='cpu', output='xdo'):
        """
        Initialize push-to-talk recorder

        Args:
            hotkey: Key to hold for recording (default: alt)
            model: Whisper model to use for transcription
            device: Device to run model on ('cpu' or 'cuda')
            output: Output method - 'xdo' or 'tmux' (default: xdo)
        """
        self.hotkey = hotkey.lower()
        self.output_method = output
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.processing_thread = None
        self.keyboard_thread = None
        self.stop_event = threading.Event()

        # Map key names to evdev key codes
        self.key_map = {
            'alt': ecodes.KEY_LEFTALT,
            'leftalt': ecodes.KEY_LEFTALT,
            'rightalt': ecodes.KEY_RIGHTALT,
            'ctrl': ecodes.KEY_LEFTCTRL,
            'leftctrl': ecodes.KEY_LEFTCTRL,
            'rightctrl': ecodes.KEY_RIGHTCTRL,
            'shift': ecodes.KEY_LEFTSHIFT,
            'leftshift': ecodes.KEY_LEFTSHIFT,
            'rightshift': ecodes.KEY_RIGHTSHIFT,
            'space': ecodes.KEY_SPACE,
            'tab': ecodes.KEY_TAB,
            'enter': ecodes.KEY_ENTER,
            'esc': ecodes.KEY_ESC,
            'capslock': ecodes.KEY_CAPSLOCK,
        }

        # Add function keys
        for i in range(1, 13):
            self.key_map[f'f{i}'] = getattr(ecodes, f'KEY_F{i}')

        # Get the key code for the hotkey
        self.target_key = self.key_map.get(self.hotkey, None)
        if self.target_key is None:
            print(f"Warning: Unknown hotkey '{hotkey}', using left Alt as default")
            self.target_key = ecodes.KEY_LEFTALT
            self.hotkey = 'alt'

        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Find keyboard devices
        self.keyboards = find_keyboard_devices()
        if not self.keyboards:
            print("Error: No keyboard devices found.")
            print("Make sure you have permission to access /dev/input/event*")
            print("You may need to run: sudo usermod -a -G input $USER")
            print("Then log out and log back in.")
            sys.exit(1)

        print(f"Found {len(self.keyboards)} keyboard device(s)")

        # Load Whisper model
        print(f"Loading Whisper model '{model}'...")
        self.model = WhisperModel(model, device=device, compute_type='int8' if device == 'cpu' else 'float16')
        print(f"Model loaded. Press and hold [{self.hotkey.upper()}] to record, release to transcribe.")

        # Start keyboard monitoring thread
        self.keyboard_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self.keyboard_thread.start()

    def _monitor_keyboard(self):
        """Monitor keyboard events"""
        key_pressed = False

        while not self.stop_event.is_set():
            # Create a dict to monitor all keyboard devices
            devices = {dev.fd: dev for dev in self.keyboards}

            # Use select to wait for events
            r, w, x = select.select(devices, [], [], 0.1)

            for fd in r:
                device = devices[fd]
                try:
                    for event in device.read():
                        if event.type == ecodes.EV_KEY:
                            key_event = categorize(event)

                            # Check if it's our target key
                            if key_event.scancode == self.target_key:
                                if key_event.keystate == 1 and not key_pressed:  # Key pressed
                                    key_pressed = True
                                    self._on_key_press()
                                elif key_event.keystate == 0 and key_pressed:  # Key released
                                    key_pressed = False
                                    self._on_key_release()
                except (OSError, IOError):
                    # Device might have been disconnected
                    continue

    def _on_key_press(self):
        """Handle key press event"""
        if not self.recording:
            self.recording = True
            self.audio_queue = queue.Queue()  # Clear any old data
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            sys.stderr.write("\r[Recording...] Press and hold to continue")
            sys.stderr.flush()

    def _on_key_release(self):
        """Handle key release event"""
        if self.recording:
            self.recording = False
            sys.stderr.write("\r" + " " * 50 + "\r")  # Clear the recording message
            sys.stderr.flush()
            # Wait for recording thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=0.5)
            # Start processing in background
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.start()

    def _record_audio(self):
        """Record audio while key is held"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            # Record audio chunks while recording flag is True
            while self.recording:
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    print(f"\nError recording audio: {e}")
                    break

            # Close stream
            self.stream.stop_stream()
            self.stream.close()

        except Exception as e:
            print(f"\nFailed to open audio stream: {e}")
            self.recording = False

    def _process_audio(self):
        """Process recorded audio and send transcription to tmux"""
        # Collect all audio chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        if not audio_chunks:
            return

        # Convert audio data to numpy array
        audio_data = b''.join(audio_chunks)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Check if audio is too short (less than 0.1 seconds)
        if len(audio_array) < self.RATE * 0.1:
            return

        # Transcribe audio
        try:
            sys.stderr.write("[Processing...]")
            sys.stderr.flush()

            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language='en',
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect transcription
            transcription = ' '.join([segment.text.strip() for segment in segments]).strip()

            sys.stderr.write("\r" + " " * 50 + "\r")  # Clear processing message
            sys.stderr.flush()

            if transcription:
                # Print to stdout for logging
                print(transcription, flush=True)

                # Send to appropriate output
                if self.output_method == 'tmux':
                    send_to_tmux(transcription + ' ')
                else:  # xdo
                    send_to_xdo(transcription + ' ')

        except Exception as e:
            print(f"\nError during transcription: {e}")

    def run(self):
        """Run the recorder main loop"""
        print(f"Push-to-talk ready. Hold [{self.hotkey.upper()}] to record.")
        print("Press Ctrl+C to exit.")

        try:
            # Keep the main thread alive
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.recording = False
        self.stop_event.set()

        # Wait for threads to finish
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

        # Close keyboard devices
        for device in self.keyboards:
            try:
                device.close()
            except:
                pass

        # Close audio resources
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        self.audio.terminate()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Push-to-talk speech transcription with XDO/tmux output')
    parser.add_argument('--hotkey', default='alt', help='Hotkey to hold for recording (default: alt)')
    parser.add_argument('--model', default='base.en', help='Whisper model to use (default: base.en)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to run on (default: cpu)')
    parser.add_argument('--output', default='xdo', choices=['xdo', 'tmux'], help='Output method (default: xdo)')

    args = parser.parse_args()

    # Check if user has permissions
    try:
        test_devices = find_keyboard_devices()
        if not test_devices:
            print("\nError: Cannot access keyboard devices.")
            print("Please run the following command to add yourself to the input group:")
            print("  sudo usermod -a -G input $USER")
            print("Then log out and log back in for the changes to take effect.")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking keyboard access: {e}")
        sys.exit(1)

    recorder = PushToTalkRecorder(
        hotkey=args.hotkey,
        model=args.model,
        device=args.device,
        output=args.output
    )

    recorder.run()
