import re
from typing import Tuple, Union, Dict, List, Any

def parse_reference_range(range_str: str) -> Tuple[float, float]:
    """
    Parse reference range string into min and max values.
    Handles formats like "3.5-5.5", "< 200", "> 40", etc.
    
    Args:
        range_str: String containing the reference range
        
    Returns:
        Tuple of (min_value, max_value)
    """
    # Clean the string
    range_str = range_str.strip()
    
    # Case 1: Range format "X-Y"
    if "-" in range_str and not ("<" in range_str or ">" in range_str):
        parts = range_str.split("-")
        if len(parts) == 2:
            try:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return (min_val, max_val)
            except ValueError:
                # Handle case where we can't convert to float
                pass
    
    # Case 2: Less than format "< X" or "<= X"
    if "<" in range_str:
        match = re.search(r"<\s*=?\s*(\d+\.?\d*)", range_str)
        if match:
            try:
                max_val = float(match.group(1))
                return (float("-inf"), max_val)
            except ValueError:
                pass
    
    # Case 3: Greater than format "> X" or ">= X"
    if ">" in range_str:
        match = re.search(r">\s*=?\s*(\d+\.?\d*)", range_str)
        if match:
            try:
                min_val = float(match.group(1))
                return (min_val, float("inf"))
            except ValueError:
                pass
    
    # Default case when parsing fails
    return (0.0, 0.0)

def is_value_out_of_range(value: float, reference_range: str) -> bool:
    """
    Check if a value is outside the reference range.
    
    Args:
        value: The test value
        reference_range: String containing the reference range
        
    Returns:
        Boolean indicating if the value is out of range
    """
    min_val, max_val = parse_reference_range(reference_range)
    
    # Check if value is outside range
    if min_val != float("-inf") and value < min_val:
        return True
    if max_val != float("inf") and value > max_val:
        return True
    
    return False

def clean_test_name(name: str) -> str:
    """
    Clean and standardize test names.
    
    Args:
        name: Raw test name string
        
    Returns:
        Cleaned test name
    """
    # Remove extra whitespace
    name = " ".join(name.split())
    
    
    patterns_to_remove = [
        r"\s*\([^)]*\)\s*$",  
        r"\s*-\s*Result\s*$",  
    ]
    
    for pattern in patterns_to_remove:
        name = re.sub(pattern, "", name)
    
    return name.strip()

def format_lab_test(test_name: str, value_str: str, reference_range: str) -> Dict[str, Any]:
    """
    Format a lab test into the required output format.
    
    Args:
        test_name: Name of the lab test
        value_str: String representation of the test value
        reference_range: Reference range for the test
        
    Returns:
        Formatted dictionary for the lab test
    """
    try:
        value = float(value_str)
        value_formatted = value
    except ValueError:
        value = value_str
        value_formatted = value_str

    # Extract unit from the reference range if possible
    test_unit = ""
    unit_match = re.search(r"[a-zA-Z/%]+", reference_range)
    if unit_match:
        test_unit = unit_match.group(0)

    out_of_range = False
    if isinstance(value, float):
        out_of_range = is_value_out_of_range(value, reference_range)

    return {
        "test_name": clean_test_name(test_name),
        "test_value": value_formatted,
        "bio_reference_range": reference_range,
        "test_unit": test_unit,
        "lab_test_out_of_range": out_of_range
    }