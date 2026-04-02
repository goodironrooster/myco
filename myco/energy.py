# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Energy tracking for myco.

Tracks energy expenditure per session.
Wraps pynvml for GPU energy reading, falls back to CPU time.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EnergyReading:
    """A single energy reading."""
    timestamp: str
    joules: float
    tokens: int
    joules_per_token: float
    source: str  # "gpu" or "cpu"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp,
            "joules": self.joules,
            "tokens": self.tokens,
            "joules_per_token": self.joules_per_token,
            "source": self.source,
        }


@dataclass
class SessionEnergy:
    """Energy tracking for a session."""
    start_time: str
    end_time: Optional[str] = None
    total_joules: float = 0.0
    total_tokens: int = 0
    readings: list[EnergyReading] = field(default_factory=list)
    semantic_complexity: float = 0.5
    thermal_load: float = 0.5
    
    def to_summary(self) -> str:
        """Generate session energy summary."""
        duration = "unknown"
        if self.end_time and self.start_time:
            try:
                start = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
                duration = f"{(end - start).total_seconds():.1f}s"
            except (ValueError, TypeError):
                pass
        
        jpt = self.total_joules / self.total_tokens if self.total_tokens > 0 else 0.0
        
        lines = [
            "Session Energy Summary",
            "======================",
            f"Duration: {duration}",
            f"Total joules: {self.total_joules:.2f}",
            f"Total tokens: {self.total_tokens}",
            f"Joules/token: {jpt:.6f}",
            f"Semantic complexity: {self.semantic_complexity:.2f}",
            f"Thermal load: {self.thermal_load:.2f}",
        ]
        
        return '\n'.join(lines)


class EnergyTracker:
    """Tracks energy expenditure for myco sessions.
    
    Uses pynvml for GPU energy readings if available.
    Falls back to CPU time as a proxy.
    """
    
    # Estimated energy values (joules)
    GPU_JOULES_PER_WATT_HOUR = 3600  # 1 Wh = 3600 J
    CPU_JOULES_PER_SECOND = 5.0  # Rough estimate for CPU inference
    TOKEN_JOULES_BASE = 0.0001  # Base energy per token
    
    def __init__(self):
        """Initialize the energy tracker."""
        self._nvml_available = False
        self._nvml_handle = None
        self._session: Optional[SessionEnergy] = None
        self._start_time: Optional[float] = None
        
        # Try to initialize pynvml
        self._init_nvml()
    
    def _init_nvml(self) -> None:
        """Initialize NVML for GPU energy monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
        except ImportError:
            self._nvml_available = False
        except Exception:
            self._nvml_available = False
    
    def start_session(self) -> None:
        """Start a new energy tracking session."""
        self._start_time = time.time()
        self._session = SessionEnergy(
            start_time=datetime.utcnow().isoformat() + "Z"
        )
        
        # Get initial thermal load
        if self._nvml_available:
            try:
                import pynvml
                temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                self._session.thermal_load = min(temp / 100.0, 1.0)  # Normalize to 0-1
            except Exception:
                pass
    
    def record_inference(
        self,
        tokens: int,
        log_probs: Optional[list[float]] = None
    ) -> EnergyReading:
        """Record an inference call.
        
        Args:
            tokens: Number of tokens generated
            log_probs: Log probabilities of tokens (for complexity estimation)
            
        Returns:
            EnergyReading for this inference
        """
        if self._session is None:
            self.start_session()
        
        # Estimate energy based on tokens
        if self._nvml_available:
            # GPU-based estimation
            joules = self._estimate_gpu_energy(tokens)
            source = "gpu"
        else:
            # CPU-based estimation
            elapsed = time.time() - self._start_time if self._start_time else 1.0
            joules = elapsed * self.CPU_JOULES_PER_SECOND * (tokens / 100.0)
            source = "cpu"
        
        # Update semantic complexity if log_probs provided
        if log_probs:
            # Lower average log_prob = higher uncertainty = higher complexity
            avg_log_prob = sum(log_probs) / len(log_probs)
            self._session.semantic_complexity = min(1.0, max(0.0, -avg_log_prob / 10.0))
        
        jpt = joules / tokens if tokens > 0 else 0.0
        
        reading = EnergyReading(
            timestamp=datetime.utcnow().isoformat() + "Z",
            joules=joules,
            tokens=tokens,
            joules_per_token=jpt,
            source=source
        )
        
        self._session.readings.append(reading)
        self._session.total_joules += joules
        self._session.total_tokens += tokens
        
        return reading
    
    def _estimate_gpu_energy(self, tokens: int) -> float:
        """Estimate GPU energy for token generation.
        
        Args:
            tokens: Number of tokens
            
        Returns:
            Estimated energy in joules
        """
        if not self._nvml_available:
            return tokens * self.TOKEN_JOULES_BASE
        
        try:
            import pynvml
            
            # Get current power draw
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            power_w = power_mw / 1000.0
            
            # Estimate time per token (rough approximation)
            seconds_per_token = 0.05  # 50ms per token
            
            joules = power_w * seconds_per_token * tokens
            return joules
            
        except Exception:
            return tokens * self.TOKEN_JOULES_BASE
    
    def end_session(self) -> SessionEnergy:
        """End the current energy tracking session.
        
        Returns:
            SessionEnergy with final totals
        """
        if self._session is None:
            self._session = SessionEnergy(
                start_time=datetime.utcnow().isoformat() + "Z"
            )
        
        self._session.end_time = datetime.utcnow().isoformat() + "Z"
        
        # Update thermal load at end
        if self._nvml_available:
            try:
                import pynvml
                temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                self._session.thermal_load = min(temp / 100.0, 1.0)
            except Exception:
                pass
        
        session = self._session
        self._session = None
        self._start_time = None
        
        return session
    
    def should_route_to_quantized(self) -> bool:
        """Check if inference should route to quantized model.
        
        Routes to lighter model if complexity < 0.4 and thermal_load > 0.8.
        
        Returns:
            True if should use quantized model path
        """
        if self._session is None:
            return False
        
        return (
            self._session.semantic_complexity < 0.4 and
            self._session.thermal_load > 0.8
        )
    
    def get_current_session(self) -> Optional[SessionEnergy]:
        """Get the current session energy tracking.
        
        Returns:
            SessionEnergy or None if no session active
        """
        return self._session
    
    def shutdown(self) -> None:
        """Shutdown NVML and cleanup."""
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_available = False


# Global tracker instance
_tracker: Optional[EnergyTracker] = None


def get_tracker() -> EnergyTracker:
    """Get the global energy tracker.
    
    Returns:
        EnergyTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = EnergyTracker()
    return _tracker


def record_inference(tokens: int, log_probs: Optional[list[float]] = None) -> EnergyReading:
    """Record an inference call using the global tracker.
    
    Args:
        tokens: Number of tokens generated
        log_probs: Log probabilities of tokens
        
    Returns:
        EnergyReading for this inference
    """
    return get_tracker().record_inference(tokens, log_probs)


def get_session_summary() -> Optional[str]:
    """Get the current session energy summary.
    
    Returns:
        Summary string or None if no session active
    """
    tracker = get_tracker()
    session = tracker.get_current_session()
    if session:
        return session.to_summary()
    return None
