# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Attractor detection for myco.

Detects when output has locked into an attractor basin (repetitive patterns).
Applies structured perturbations to escape.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttractorEvent:
    """Record of an attractor event."""
    name: str
    turn_detected: int
    similarity_score: float
    perturbation_applied: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for world model."""
        return {
            "name": self.name,
            "turn_detected": self.turn_detected,
            "similarity_score": self.similarity_score,
            "perturbation_applied": self.perturbation_applied,
        }


class AttractorDetector:
    """Detects attractor basins in output patterns.
    
    Monitors output embeddings and detects lock-in when cosine similarity
    exceeds 0.92 for three consecutive turns.
    """
    
    SIMILARITY_THRESHOLD = 0.92
    CONSECUTIVE_TURNS = 3
    BUFFER_SIZE = 5
    
    def __init__(self):
        """Initialize the attractor detector."""
        # Rolling buffer of output embeddings (simplified as text hashes)
        self._outputs: deque[str] = deque(maxlen=self.BUFFER_SIZE)
        self._consecutive_high_similarity = 0
        self._is_locked = False
        self._current_attractor: Optional[str] = None
        self._turn_count = 0
        self._events: list[AttractorEvent] = []
    
    def add_output(self, output: str) -> None:
        """Add a new output to the buffer.
        
        Args:
            output: The output text to track
        """
        self._turn_count += 1
        self._outputs.append(output)
        
        # Check for attractor lock-in
        if len(self._outputs) >= 2:
            similarity = self._calculate_similarity(
                self._outputs[-1],
                self._outputs[-2]
            )
            
            if similarity >= self.SIMILARITY_THRESHOLD:
                self._consecutive_high_similarity += 1
            else:
                self._consecutive_high_similarity = 0
            
            # Detect lock-in after 3 consecutive high-similarity pairs
            if self._consecutive_high_similarity >= self.CONSECUTIVE_TURNS:
                self._is_locked = True
                if not self._current_attractor:
                    self._current_attractor = self._name_attractor()
                    event = AttractorEvent(
                        name=self._current_attractor,
                        turn_detected=self._turn_count,
                        similarity_score=similarity
                    )
                    self._events.append(event)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Uses a simplified bag-of-words approach for efficiency.
        For production, this would use actual embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Tokenize and count words
        def tokenize(text: str) -> dict[str, int]:
            words = text.lower().split()
            # Remove common words and punctuation
            words = [
                w.strip('.,!?;:()[]{}"\'')
                for w in words
                if len(w) > 2
            ]
            counts: dict[str, int] = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            return counts
        
        counts1 = tokenize(text1)
        counts2 = tokenize(text2)
        
        if not counts1 or not counts2:
            return 0.0
        
        # Get all unique words
        all_words = set(counts1.keys()) | set(counts2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = 0.0
        mag1 = 0.0
        mag2 = 0.0
        
        for word in all_words:
            v1 = counts1.get(word, 0)
            v2 = counts2.get(word, 0)
            dot_product += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / ((mag1 ** 0.5) * (mag2 ** 0.5))
    
    def _name_attractor(self) -> str:
        """Generate a name for the current attractor basin.
        
        Returns:
            A descriptive name for the attractor
        """
        if not self._outputs:
            return "unknown_attractor"
        
        # Analyze the repeated pattern
        recent = list(self._outputs)[-3:]
        
        # Common attractor patterns
        if all("import" in text.lower() for text in recent):
            return "import_restructure_loop"
        if all("function" in text.lower() for text in recent):
            return "function_extraction_loop"
        if all("class" in text.lower() for text in recent):
            return "class_definition_loop"
        if all("test" in text.lower() for text in recent):
            return "test_generation_loop"
        if all("error" in text.lower() or "fix" in text.lower() for text in recent):
            return "error_fix_loop"
        
        return f"output_pattern_{self._turn_count}"
    
    def is_locked(self) -> bool:
        """Check if currently in an attractor basin.
        
        Returns:
            True if locked into a repetitive pattern
        """
        return self._is_locked
    
    def get_attractor_name(self) -> Optional[str]:
        """Get the name of the current attractor.
        
        Returns:
            Attractor name or None if not locked
        """
        return self._current_attractor
    
    def select_perturbation(self) -> str:
        """Select a perturbation to escape the attractor.
        
        Chooses the perturbation with lowest cosine similarity to recent output.
        
        Returns:
            Name of the perturbation to apply
        """
        if not self._outputs:
            return "perspective_inversion"
        
        recent = self._outputs[-1]
        
        # Define perturbation strategies
        perturbations = {
            "perspective_inversion": self._describe_perspective_inversion(),
            "constraint_removal": self._describe_constraint_removal(),
            "domain_shift": self._describe_domain_shift(),
        }
        
        # Calculate similarity to each perturbation description
        similarities = {
            name: self._calculate_similarity(recent, desc)
            for name, desc in perturbations.items()
        }
        
        # Choose the perturbation most different from recent output
        best_perturbation = min(similarities, key=similarities.get)
        
        return best_perturbation
    
    def _describe_perspective_inversion(self) -> str:
        """Describe the perspective inversion perturbation."""
        return (
            "Restate the problem from the perspective of the module being imported, "
            "not the importer. Consider what constraints the imported module needs "
            "to satisfy rather than what the importer wants from it."
        )
    
    def _describe_constraint_removal(self) -> str:
        """Describe the constraint removal perturbation."""
        return (
            "Identify the assumption most taken for granted in the current approach "
            "and drop it. What if this constraint didn't exist? How would the "
            "solution change?"
        )
    
    def _describe_domain_shift(self) -> str:
        """Describe the domain shift perturbation."""
        return (
            "Reframe the structural problem using a non-programming analogy: "
            "materials science (stress/strain), mycology (growth patterns), "
            "or fluid dynamics (flow/pressure). Then translate back to code."
        )
    
    def apply_perturbation(self, perturbation_type: str) -> str:
        """Apply a perturbation and return guidance.
        
        Args:
            perturbation_type: Type of perturbation to apply
            
        Returns:
            Guidance text for escaping the attractor
        """
        guidance = {
            "perspective_inversion": (
                "🔄 Perspective Inversion: Instead of asking what the importer needs, "
                "ask what the imported module's responsibilities should be. "
                "Design the interface from the implementer's perspective."
            ),
            "constraint_removal": (
                "🔄 Constraint Removal: The current approach assumes a constraint "
                "that may not be necessary. Identify and remove the most limiting "
                "assumption. What becomes possible?"
            ),
            "domain_shift": (
                "🔄 Domain Shift: Think of this code structure like a mycelial network. "
                "Pressure flows through the path of least resistance. Where is the "
                "structural tension building? How would natural growth resolve it?"
            ),
        }
        
        result = guidance.get(perturbation_type, "Apply a different approach.")
        
        # Record the event
        if self._events:
            self._events[-1].perturbation_applied = perturbation_type
        
        # Reset lock state
        self._is_locked = False
        self._consecutive_high_similarity = 0
        self._current_attractor = None
        
        return result
    
    def get_events(self) -> list[AttractorEvent]:
        """Get all attractor events detected.
        
        Returns:
            List of attractor events
        """
        return self._events.copy()
    
    def reset(self) -> None:
        """Reset the detector state."""
        self._outputs.clear()
        self._consecutive_high_similarity = 0
        self._is_locked = False
        self._current_attractor = None
        self._turn_count = 0
