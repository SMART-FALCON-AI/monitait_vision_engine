from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os
import time
import threading
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STATE MANAGEMENT CLASSES (inspired by fabriqc-local-server)
# =============================================================================

class CaptureState(Enum):
    """Capture states for camera triggering."""
    IDLE = "idle"              # No capture pending
    READY = "ready"            # Ready to capture on trigger
    CAPTURING = "capturing"    # Currently capturing
    PROCESSING = "processing"  # Processing captured images
    ERROR = "error"           # Error state


@dataclass
class CapturePhase:
    """Represents a single capture phase (light mode + delay + cameras + trigger thresholds).

    A capture sequence can have multiple phases, each with different light settings.
    Example: Phase 1 (U light) captures cam 1,2,3, Phase 2 (B light) captures cam 4.

    steps threshold:
      -1 = infinite loop (always capture, no encoder check)
       1 = capture on every 1 step change (default)
       N = capture on every N step changes
    """
    light_mode: str = "U_ON_B_OFF"  # Light mode: U_ON_B_OFF, U_OFF_B_ON, U_ON_B_ON, U_OFF_B_OFF
    delay: float = 0.13  # Delay in seconds after setting light before capture
    cameras: List[int] = field(default_factory=lambda: [1, 2, 3])  # Camera IDs to capture in this phase
    steps: int = 1  # Step/encoder count threshold (1 = every step, -1 = infinite loop)
    analog: int = -1  # Analog value threshold (-1 = disabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "light_mode": self.light_mode,
            "delay": self.delay,
            "cameras": self.cameras,
            "steps": self.steps,
            "analog": self.analog
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapturePhase':
        return cls(
            light_mode=data.get('light_mode', 'U_ON_B_OFF'),
            delay=float(data.get('delay', 0.13)),
            cameras=data.get('cameras', [1, 2, 3]),
            steps=int(data.get('steps', 1)),  # Default: 1 step change
            analog=int(data.get('analog', -1))
        )


def _get_default_cameras() -> List[int]:
    """Get default camera IDs based on detected cameras."""
    try:
        from services.camera import DETECTED_CAMERAS
        num_cameras = len(DETECTED_CAMERAS) if DETECTED_CAMERAS else 4
    except ImportError:
        num_cameras = 4
    return list(range(1, num_cameras + 1))


@dataclass
class State:
    """Represents a capture state configuration with multiple phases.

    This class defines a complete capture sequence:
    - Multiple capture phases (each with light mode, delay, cameras)
    - Whether state is enabled
    - Trigger thresholds (steps and analog) for when to activate

    Default: Single phase with U_ON_B_OFF light, 0.1s delay, all detected cameras.
    """
    name: str
    phases: List[CapturePhase] = field(default_factory=lambda: [
        CapturePhase(light_mode="U_ON_B_OFF", delay=0.1, cameras=_get_default_cameras())
    ])
    enabled: bool = True
    # Trigger thresholds (kept for backward compatibility, prefer phase-level thresholds)
    steps: int = 1  # Step/encoder count threshold (1 = every step, -1 = infinite loop)
    analog: int = -1  # Analog value threshold (-1 = disabled)

    def should_trigger(self, encoder_value: int, analog_value: int) -> bool:
        """Check if state trigger conditions are met.

        Args:
            encoder_value: Current encoder/step count
            analog_value: Current analog sensor value

        Returns:
            True if both thresholds are met (or disabled), False otherwise
        """
        steps_met = self.steps < 0 or encoder_value >= self.steps
        analog_met = self.analog < 0 or analog_value >= self.analog
        return steps_met and analog_met

    def get_all_cameras(self) -> List[int]:
        """Get all cameras used across all phases."""
        cameras = []
        for phase in self.phases:
            cameras.extend(phase.cameras)
        return list(set(cameras))

    def is_camera_active(self, camera_id: int) -> bool:
        """Check if a specific camera is active in any phase."""
        return camera_id in self.get_all_cameras()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "phases": [p.to_dict() for p in self.phases],
            "enabled": self.enabled,
            "steps": self.steps,
            "analog": self.analog
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create a State instance from a dictionary."""
        phases_data = data.get('phases', [])
        if phases_data:
            phases = [CapturePhase.from_dict(p) for p in phases_data]
        else:
            # Legacy format support - convert old single-phase format
            phases = [CapturePhase(
                light_mode=data.get('light_mode', 'U_ON_B_OFF'),
                delay=float(data.get('delay', 0.13)),
                cameras=data.get('active_cameras', [1, 2, 3, 4])
            )]

        return cls(
            name=data.get('name', 'default'),
            phases=phases,
            enabled=data.get('enabled', True),
            steps=int(data.get('steps', 1)),
            analog=int(data.get('analog', -1))
        )


class StateManager:
    """Manages state transitions and camera capture synchronization.

    This class handles:
    - State transitions based on encoder and analog values
    - Camera synchronization - which cameras are active
    - Light mode changes
    - Capture timing and delays
    """

    def __init__(self, watcher=None):
        self.watcher = watcher
        self.current_state: Optional[State] = None
        self.capture_state: CaptureState = CaptureState.IDLE
        self.states: Dict[str, State] = {}  # Named states
        self.last_light_mode: Optional[str] = None

        # State synchronization
        self.state_lock = threading.Lock()
        self.state_ready = threading.Event()
        self.capture_complete = threading.Event()

        # Camera-specific events
        self.camera_ready: Dict[int, threading.Event] = {}
        self.camera_complete: Dict[int, threading.Event] = {}

        # Metrics
        self.last_error: Optional[str] = None
        self.error_time: Optional[float] = None
        self.capture_count: int = 0
        self.last_capture_time: Optional[float] = None

        # Initialize default state
        self._init_default_states()

    def _init_default_states(self):
        """Initialize default states using all detected cameras.

        Creates a simple default state that captures all available cameras
        with a single phase (U_ON_B_OFF light, 0.1s delay).
        Steps=-1 means capture on any encoder change.
        """
        # Determine available camera IDs based on watcher's initialized cameras
        if self.watcher and hasattr(self.watcher, 'camera_paths') and self.watcher.camera_paths:
            num_cameras = len(self.watcher.camera_paths)
            all_camera_ids = list(range(1, num_cameras + 1))
        else:
            # No cameras available
            num_cameras = 0
            all_camera_ids = []

        if num_cameras > 0:
            logger.info(f"Initializing default state with {num_cameras} camera(s): {all_camera_ids}")
        else:
            logger.info("Initializing default state with no cameras. Add cameras via IP Camera Discovery.")

        # Default state: Single phase capturing all cameras with 0.1s delay
        # steps=1 means trigger on every 1 step change (default)
        self.states["default"] = State(
            name="default",
            phases=[
                CapturePhase(
                    light_mode="U_ON_B_OFF",
                    delay=0.1,
                    cameras=all_camera_ids,
                    steps=1,  # 1 = capture on every step change
                    analog=-1  # -1 = no analog threshold
                )
            ],
            enabled=True
        )
        self.current_state = self.states["default"]

    def add_state(self, state: State) -> bool:
        """Add or update a named state."""
        try:
            with self.state_lock:
                self.states[state.name] = state
                logger.info(f"Added/updated state: {state.name}")
                return True
        except Exception as e:
            self._handle_error(f"Error adding state: {e}")
            return False

    def remove_state(self, name: str) -> bool:
        """Remove a named state."""
        try:
            with self.state_lock:
                if name == "default":
                    logger.warning("Cannot remove default state")
                    return False
                if name in self.states:
                    del self.states[name]
                    logger.info(f"Removed state: {name}")
                    return True
                return False
        except Exception as e:
            self._handle_error(f"Error removing state: {e}")
            return False

    def set_current_state(self, name: str) -> bool:
        """Set the current active state by name."""
        try:
            with self.state_lock:
                if name not in self.states:
                    logger.error(f"State not found: {name}")
                    return False

                state = self.states[name]

                # Handle light mode change for first phase if needed
                if state.phases:
                    first_phase_mode = state.phases[0].light_mode
                    if first_phase_mode != self.last_light_mode:
                        self._handle_light_mode_change(first_phase_mode)
                    self.last_light_mode = first_phase_mode

                self.current_state = state
                self.state_ready.set()
                logger.info(f"Set current state to: {name}")
                return True
        except Exception as e:
            self._handle_error(f"Error setting state: {e}")
            return False

    def _handle_light_mode_change(self, new_mode: str) -> None:
        """Handle light mode changes via watcher commands."""
        if self.watcher is None:
            return
        try:
            # Map light mode to watcher command
            mode_commands = {
                "U_ON_B_OFF": "U_ON_B_OFF",
                "U_OFF_B_ON": "B_ON_U_OFF",
                "U_ON_B_ON": "U_ON_B_ON",
                "U_OFF_B_OFF": "B_OFF_U_OFF"
            }
            if new_mode in mode_commands:
                cmd = mode_commands[new_mode]
                # Use watcher's light control methods
                if hasattr(self.watcher, 'send_command'):
                    self.watcher.send_command(cmd)
                logger.info(f"Changed light mode to: {new_mode}")
        except Exception as e:
            self._handle_error(f"Error changing light mode: {e}")

    def _handle_error(self, error_msg: str) -> None:
        """Handle errors consistently."""
        self.last_error = error_msg
        self.error_time = time.time()
        self.capture_state = CaptureState.ERROR
        logger.error(error_msg)

    def is_camera_active(self, camera_id: int) -> bool:
        """Check if a camera is active in the current state."""
        if self.current_state is None:
            return True  # Default to active if no state set
        return self.current_state.is_camera_active(camera_id)

    def should_capture(self, encoder_value: int = 0, analog_value: int = 0) -> bool:
        """Check if capture should be triggered based on current state.

        Args:
            encoder_value: Current encoder/step count since last capture
            analog_value: Current analog sensor value

        Returns:
            True if capture should be triggered
        """
        if self.current_state is None:
            return True  # Default to always capture if no state
        if not self.current_state.enabled:
            return False
        if len(self.current_state.phases) == 0:
            return False
        # Check trigger thresholds (steps and analog)
        return self.current_state.should_trigger(encoder_value, analog_value)

    def trigger_capture(self) -> bool:
        """Trigger a capture cycle."""
        try:
            with self.state_lock:
                if self.capture_state == CaptureState.CAPTURING:
                    logger.warning("Capture already in progress")
                    return False

                self.capture_state = CaptureState.CAPTURING
                self.capture_complete.clear()

                # Note: Delays are handled per-phase in capture_frames(), not here

                self.capture_count += 1
                self.last_capture_time = time.time()
                return True
        except Exception as e:
            self._handle_error(f"Error triggering capture: {e}")
            return False

    def complete_capture(self) -> None:
        """Signal that capture is complete."""
        with self.state_lock:
            self.capture_state = CaptureState.IDLE
            self.capture_complete.set()

    def get_status(self) -> Dict[str, Any]:
        """Get current state manager status."""
        return {
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "capture_state": self.capture_state.value,
            "states": {name: s.to_dict() for name, s in self.states.items()},
            "capture_count": self.capture_count,
            "last_capture_time": self.last_capture_time,
            "last_error": self.last_error,
            "error_time": self.error_time
        }

    def save_states(self, filepath: str) -> bool:
        """Save all states to a file."""
        try:
            data = {name: s.to_dict() for name, s in self.states.items()}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved states to {filepath}")
            return True
        except Exception as e:
            self._handle_error(f"Error saving states: {e}")
            return False

    def load_states(self, filepath: str) -> bool:
        """Load states from a file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"States file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                data = json.load(f)

            for name, state_data in data.items():
                self.states[name] = State.from_dict(state_data)

            logger.info(f"Loaded {len(data)} states from {filepath}")
            return True
        except Exception as e:
            self._handle_error(f"Error loading states: {e}")
            return False


# Global state manager instance (initialized after watcher)
state_manager: Optional[StateManager] = None
