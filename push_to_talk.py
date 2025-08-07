#!/usr/bin/env python3
import os
import sys
import warnings
import subprocess
import threading
import time
import queue
import select
import logging
from enum import Enum

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

# Processing state enumeration for clean state management
class RecorderState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    CANCELLED = "cancelled"

# Constants for configuration and maintainability
class Config:
    # Audio settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    MIN_AUDIO_DURATION = 0.1  # seconds

    # Timeout settings
    THREAD_JOIN_TIMEOUT = 2.0  # seconds
    RECORDING_THREAD_TIMEOUT = 0.5  # seconds
    SUBPROCESS_TIMEOUT = 1.0  # seconds

    # User messages
    MSG_RECORDING = "[Recording...] Press and hold to continue"
    MSG_PROCESSING = "[Processing...]"
    MSG_CANCELLED = "[Cancelled]"
    MSG_CLEAR_LINE = "\r" + " " * 50 + "\r"

    # Whisper settings
    WHISPER_BEAM_SIZE = 5
    WHISPER_LANGUAGE = 'en'
    WHISPER_MIN_SILENCE_MS = 500

def send_to_tmux(text):
    """Send text to the currently focused tmux pane"""
    try:
        # Get currently focused pane
        result = subprocess.run(
            ['tmux', 'display-message', '-p', '#{pane_id}'],
            capture_output=True,
            text=True,
            timeout=Config.SUBPROCESS_TIMEOUT
        )

        if result.returncode == 0:
            pane_id = result.stdout.strip()
            # Send text to the focused pane
            subprocess.run(
                ['tmux', 'send-keys', '-t', pane_id, text],
                timeout=Config.SUBPROCESS_TIMEOUT
            )
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Tmux not available or command failed
        pass
    return False

def send_to_xdo(text):
    """Send text to the currently focused window using custom XDO implementation"""
    try:
        # Import custom XDO module
        from custom_xdo import xdo_type

        # Use custom XDO implementation with timeout
        return xdo_type(text, timeout=Config.SUBPROCESS_TIMEOUT)

    except ImportError:
        # Fall back to subprocess xdotool if custom implementation not available
        try:
            subprocess.run(
                ['xdotool', 'type', '--', text],
                timeout=Config.SUBPROCESS_TIMEOUT
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False
    except Exception:
        # Custom XDO failed, try subprocess fallback
        try:
            subprocess.run(
                ['xdotool', 'type', '--', text],
                timeout=Config.SUBPROCESS_TIMEOUT
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
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

class UserFeedback:
    """Centralized user feedback system for clean logging"""

    @staticmethod
    def show_message(message, clear_line=False):
        """Display a message to the user"""
        if clear_line:
            sys.stderr.write(Config.MSG_CLEAR_LINE)
        sys.stderr.write(f"\r{message}")
        sys.stderr.flush()

    @staticmethod
    def clear_message():
        """Clear the current message line"""
        sys.stderr.write(Config.MSG_CLEAR_LINE)
        sys.stderr.flush()

    @staticmethod
    def print_status(message):
        """Print status to stdout for logging"""
        print(message, flush=True)

class PushToTalkRecorder:
    def __init__(self, hotkey='alt', model='base.en', device='cpu', output='xdo'):
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

        # State management using enumeration
        self.state = RecorderState.IDLE
        self.state_lock = threading.Lock()

        # Thread-safe cancellation flag
        self.cancel_processing = threading.Event()

        # Audio and thread management
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

        # Audio settings from Config
        self.FORMAT = Config.FORMAT
        self.CHANNELS = Config.CHANNELS
        self.RATE = Config.RATE
        self.CHUNK = Config.CHUNK

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

        # Initialize key tracking instance variable
        self.key_pressed = False

        # Load Whisper model
        print(f"Loading Whisper model '{model}'...")
        self.model = WhisperModel(model, device=device, compute_type='int8' if device == 'cpu' else 'float16')
        print(f"Model loaded. Press and hold [{self.hotkey.upper()}] to record, release to transcribe.")

        # Start keyboard monitoring thread
        self.keyboard_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self.keyboard_thread.start()

    def _set_state(self, new_state):
        """Thread-safe state setter"""
        with self.state_lock:
            self.state = new_state

    def _get_state(self):
        """Thread-safe state getter"""
        with self.state_lock:
            return self.state

    def _is_state(self, state):
        """Thread-safe state checker"""
        with self.state_lock:
            return self.state == state

    def _handle_key_press_event(self):
        """Handle Alt key press based on current state"""
        current_state = self._get_state()

        if current_state == RecorderState.IDLE:
            self._start_recording()
        elif current_state == RecorderState.PROCESSING:
            # Cancel current processing and immediately start new recording
            self._cancel_processing(start_new_recording=True)
            self._start_recording()

    def _handle_key_release_event(self):
        """Handle Alt key release based on current state"""
        current_state = self._get_state()

        if current_state == RecorderState.RECORDING:
            self._stop_recording_and_process()

    def _start_recording(self):
        """Start audio recording"""
        # Clean up existing recording thread if it exists and is still running
        if self.recording_thread is not None and self.recording_thread.is_alive():
            # Wait for the old recording thread to terminate
            self.recording_thread.join(timeout=Config.RECORDING_THREAD_TIMEOUT)

        # Clean up existing audio stream if it exists and is active
        if self.stream is not None:
            try:
                if hasattr(self.stream, '_is_active') and self.stream._is_active:
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception:
                # Stream cleanup failed, but we can still proceed
                pass

        self._set_state(RecorderState.RECORDING)
        # Ensure audio queue is completely clear for new recording
        self.audio_queue = queue.Queue()  # Clear any old data
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        UserFeedback.show_message(Config.MSG_RECORDING)

    def _stop_recording_and_process(self):
        """Stop recording and start processing"""
        self._set_state(RecorderState.PROCESSING)
        UserFeedback.clear_message()

        # Reset key_pressed flag to ensure it's ready for next Alt press during processing
        self.key_pressed = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=Config.RECORDING_THREAD_TIMEOUT)

        # Clear cancellation flag and start processing
        self.cancel_processing.clear()
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()

    def _cancel_processing(self, start_new_recording=False):
        """Cancel ongoing processing"""
        self.cancel_processing.set()

        # Clear the audio queue to prevent old data from interfering
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Only set state to CANCELLED if we're not starting a new recording
        if not start_new_recording:
            self._set_state(RecorderState.CANCELLED)
            UserFeedback.show_message(Config.MSG_CANCELLED, clear_line=True)
        else:
            # Clear the current message for smooth transition to recording
            UserFeedback.clear_message()

    def _monitor_keyboard(self):
        """Monitor keyboard events with improved state-based handling"""

        while not self.stop_event.is_set():
            devices = {dev.fd: dev for dev in self.keyboards}
            r, w, x = select.select(devices, [], [], 0.1)

            for fd in r:
                device = devices[fd]
                try:
                    for event in device.read():
                        if event.type == ecodes.EV_KEY:
                            key_event = categorize(event)
                            if key_event.scancode == self.target_key:
                                current_state = self._get_state()
                                if key_event.keystate == 1:  # Key pressed
                                    # During PROCESSING, allow Alt press regardless of key_pressed flag
                                    if current_state == RecorderState.PROCESSING or not self.key_pressed:
                                        self.key_pressed = True
                                        self._handle_key_press_event()
                                elif key_event.keystate == 0 and self.key_pressed:  # Key released
                                    self.key_pressed = False
                                    self._handle_key_release_event()
                except (OSError, IOError):
                    continue

    def _record_audio(self):
        """Record audio while in recording state"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            # Record audio chunks while in recording state
            while self._is_state(RecorderState.RECORDING):
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    UserFeedback.print_status(f"Error recording audio: {e}")
                    break

            # Close stream
            self.stream.stop_stream()
            self.stream.close()

        except Exception as e:
            UserFeedback.print_status(f"Failed to open audio stream: {e}")
            self._set_state(RecorderState.IDLE)

    def _process_audio(self):
        """Process recorded audio with cancellation support"""
        try:
            # Check for early cancellation
            if self.cancel_processing.is_set():
                self._handle_processing_cancelled()
                return

            # Collect all audio chunks
            audio_chunks = []
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())

            if not audio_chunks:
                self._set_state(RecorderState.IDLE)
                return

            # Convert audio data to numpy array
            audio_data = b''.join(audio_chunks)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Check if audio is too short
            if len(audio_array) < self.RATE * Config.MIN_AUDIO_DURATION:
                self._set_state(RecorderState.IDLE)
                return

            # Check for cancellation before processing
            if self.cancel_processing.is_set():
                self._handle_processing_cancelled()
                return

            # Show processing message
            UserFeedback.show_message(Config.MSG_PROCESSING)

            # Transcribe audio
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=Config.WHISPER_BEAM_SIZE,
                language=Config.WHISPER_LANGUAGE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=Config.WHISPER_MIN_SILENCE_MS)
            )

            # Check for cancellation after transcription
            if self.cancel_processing.is_set():
                self._handle_processing_cancelled()
                return

            # Collect transcription
            transcription = ' '.join([segment.text.strip() for segment in segments]).strip()

            # Clear processing message
            UserFeedback.clear_message()

            if transcription:
                # Log transcription
                UserFeedback.print_status(transcription)

                # Check for cancellation one more time before sending output
                if self.cancel_processing.is_set():
                    self._handle_processing_cancelled()
                    return

                # Send to appropriate output
                success = False
                if self.output_method == 'tmux':
                    success = send_to_tmux(transcription + ' ')
                else:  # xdo
                    success = send_to_xdo(transcription + ' ')

                if not success:
                    UserFeedback.print_status(f"Warning: Failed to send text via {self.output_method}")

            # Reset to idle state
            self._set_state(RecorderState.IDLE)

        except Exception as e:
            self._handle_processing_error(e)

    def _handle_processing_cancelled(self):
        """Handle cancellation during processing"""
        UserFeedback.clear_message()

        # Only set state to IDLE if we're not already recording a new session
        # This prevents race condition where old processing thread kills new recording
        current_state = self._get_state()
        if current_state != RecorderState.RECORDING:
            UserFeedback.print_status("Transcription cancelled by user")
            self._set_state(RecorderState.IDLE)
        # If already RECORDING, don't change state or show cancellation message
        # The new recording should continue uninterrupted

    def _handle_processing_error(self, error):
        """Handle errors during processing"""
        UserFeedback.clear_message()
        UserFeedback.print_status(f"Error during transcription: {error}")

        # Only set state to IDLE if we're not already recording a new session
        # This prevents race condition where old processing thread kills new recording
        current_state = self._get_state()
        if current_state != RecorderState.RECORDING:
            self._set_state(RecorderState.IDLE)

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
        # Set cancellation and stop flags
        self.cancel_processing.set()
        self.stop_event.set()
        self._set_state(RecorderState.IDLE)

        # Wait for threads to finish with proper timeouts
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)

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
