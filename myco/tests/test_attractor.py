"""Tests for myco.attractor module."""

import pytest
from myco.attractor import AttractorDetector, AttractorEvent


class TestAttractorEvent:
    """Tests for AttractorEvent dataclass."""
    
    def test_create_event(self):
        """Test creating an attractor event."""
        event = AttractorEvent(
            name="test_attractor",
            turn_detected=5,
            similarity_score=0.95
        )
        
        assert event.name == "test_attractor"
        assert event.turn_detected == 5
        assert event.similarity_score == 0.95
        assert event.perturbation_applied is None
    
    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = AttractorEvent(
            name="test",
            turn_detected=3,
            similarity_score=0.93,
            perturbation_applied="perspective_inversion"
        )
        
        d = event.to_dict()
        
        assert d["name"] == "test"
        assert d["turn_detected"] == 3
        assert d["perturbation_applied"] == "perspective_inversion"


class TestAttractorDetector:
    """Tests for AttractorDetector class."""
    
    def test_create_detector(self):
        """Test creating detector with default values."""
        detector = AttractorDetector()
        
        assert detector.is_locked() is False
        assert detector.get_attractor_name() is None
        assert detector.get_events() == []
    
    def test_add_output(self):
        """Test adding outputs to detector."""
        detector = AttractorDetector()
        
        detector.add_output("test output 1")
        detector.add_output("test output 2")
        
        assert detector.is_locked() is False
    
    def test_detect_attractor_lock_in(self):
        """Test detecting attractor lock-in with similar outputs."""
        detector = AttractorDetector()
        
        # Add very similar outputs to trigger lock-in
        similar_output = "This is a test output with some repeated content"
        for _ in range(5):
            detector.add_output(similar_output)
        
        # Should detect lock-in after 3 consecutive high-similarity pairs
        assert detector.is_locked() is True
        assert detector.get_attractor_name() is not None
    
    def test_no_lock_in_with_diverse_outputs(self):
        """Test no lock-in with diverse outputs."""
        detector = AttractorDetector()
        
        diverse_outputs = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "The sky is blue today",
            "Machine learning models need data",
            "Testing is important for software quality"
        ]
        
        for output in diverse_outputs:
            detector.add_output(output)
        
        assert detector.is_locked() is False
    
    def test_select_perturbation(self):
        """Test selecting a perturbation."""
        detector = AttractorDetector()
        
        perturbation = detector.select_perturbation()
        
        assert perturbation in [
            "perspective_inversion",
            "constraint_removal",
            "domain_shift"
        ]
    
    def test_apply_perturbation_resets_lock(self):
        """Test applying perturbation resets lock state."""
        detector = AttractorDetector()
        
        # Trigger lock-in
        similar = "repeated output content"
        for _ in range(5):
            detector.add_output(similar)
        
        assert detector.is_locked() is True
        
        # Apply perturbation
        perturbation = detector.select_perturbation()
        detector.apply_perturbation(perturbation)
        
        assert detector.is_locked() is False
        assert detector.get_attractor_name() is None
    
    def test_apply_perturbation_returns_guidance(self):
        """Test applying perturbation returns guidance."""
        detector = AttractorDetector()
        
        guidance = detector.apply_perturbation("perspective_inversion")
        
        assert "Perspective Inversion" in guidance
    
    def test_get_events_records_history(self):
        """Test that events are recorded."""
        detector = AttractorDetector()
        
        # Trigger lock-in
        similar = "repeated content here"
        for _ in range(5):
            detector.add_output(similar)
        
        # Apply perturbation
        perturbation = detector.select_perturbation()
        detector.apply_perturbation(perturbation)
        
        events = detector.get_events()
        
        assert len(events) >= 1
        assert events[0].perturbation_applied is not None
    
    def test_reset_clears_state(self):
        """Test reset clears all state."""
        detector = AttractorDetector()
        
        # Add some outputs
        detector.add_output("test 1")
        detector.add_output("test 2")
        
        detector.reset()
        
        assert detector.is_locked() is False
        assert detector.get_attractor_name() is None
        assert detector.get_events() == []
    
    def test_similarity_calculation_same_text(self):
        """Test similarity calculation with identical texts."""
        detector = AttractorDetector()
        
        text = "This is exactly the same text"
        similarity = detector._calculate_similarity(text, text)
        
        # Should be very close to 1.0 (floating point precision)
        assert similarity > 0.99
    
    def test_similarity_calculation_different_text(self):
        """Test similarity calculation with different texts."""
        detector = AttractorDetector()
        
        text1 = "completely different words apple banana"
        text2 = "other words orange grape"
        
        similarity = detector._calculate_similarity(text1, text2)
        
        # Should be low but not necessarily 0
        assert similarity < 0.5
    
    def test_similarity_empty_text(self):
        """Test similarity with empty text."""
        detector = AttractorDetector()
        
        similarity = detector._calculate_similarity("", "some text")
        
        assert similarity == 0.0
    
    def test_attractor_naming_import_loop(self):
        """Test attractor naming detects import loops."""
        detector = AttractorDetector()
        
        import_text = "We need to import this module and fix the imports"
        for _ in range(5):
            detector.add_output(import_text)
        
        name = detector.get_attractor_name()
        
        assert "import" in name.lower()
    
    def test_attractor_naming_function_loop(self):
        """Test attractor naming detects function loops."""
        detector = AttractorDetector()
        
        func_text = "This function needs to be a function in the code"
        for _ in range(5):
            detector.add_output(func_text)
        
        name = detector.get_attractor_name()
        
        assert "function" in name.lower()
