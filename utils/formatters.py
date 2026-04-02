"""Utility functions for data formatting."""
from typing import Any


def format_value(value: Any) -> str:
    """Format any value as a string representation.
    
    Args:
        value: Any value to format.
        
    Returns:
        String representation of the value.
    """
    if value is None:
        return "None"
    return str(value)
