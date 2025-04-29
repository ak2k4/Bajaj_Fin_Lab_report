import re
from typing import List, Dict, Any, Optional
from app.utils import format_lab_test

class LabTestParser:
    def __init__(self):
        """Initialize the lab test parser with patterns for different lab report formats."""
        # Common patterns for lab test data
        self.patterns = [
            # Pattern 1: Test Name: Value (Range)
            r"([A-Za-z0-9\s\-\+\/]+)\s*:\s*([0-9\.]+)\s*(?:\(([0-9\.<>\-\s\.]+)\))?",
            # Pattern 2: Test Name Value Range
            r"([A-Za-z0-9\s\-\+\/]+)\s+([0-9\.]+)\s+([0-9\.<>\-\s\.]+)",
            # Pattern 3: Test Name......Value......Range
            r"([A-Za-z0-9\s\-\+\/]+)\.{2,}\s*([0-9\.]+)\.{2,}\s*([0-9\.<>\-\s\.]+)",
            # Pattern 4: Test Name Result: Value Range
            r"([A-Za-z0-9\s\-\+\/]+)\s+Result\s*:\s*([0-9\.]+)\s+([0-9\.<>\-\s\.]+)"
        ]
        
        # Patterns for table-structured data
        self.table_header_patterns = [
            r"test\s*name",
            r"parameter",
            r"investigation",
            r"lab\s*test"
        ]
        
        self.value_header_patterns = [
            r"result",
            r"value",
            r"reading"
        ]
        
        self.range_header_patterns = [
            r"reference\s*range",
            r"normal\s*range",
            r"bio\s*reference",
            r"expected\s*range"
        ]
    
    def parse_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse lab test information from extracted text.
        
        Args:
            text: Extracted text from the lab report
            
        Returns:
            List of dictionaries containing lab test information
        """
        lab_tests = []
        
        # Check if the text appears to be in tabular format
        if self._is_tabular_format(text):
            return self._parse_tabular_format(text)
        
        # Try each pattern
        for pattern in self.patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    test_name = match.group(1).strip()
                    value_str = match.group(2).strip()
                    
                    # Get reference range if available
                    reference_range = "N/A"
                    if len(match.groups()) >= 3 and match.group(3):
                        reference_range = match.group(3).strip()
                    
                    # Format the lab test
                    lab_test = format_lab_test(test_name, value_str, reference_range)
                    lab_tests.append(lab_test)
        
        # If we found tests, return them
        if lab_tests:
            return lab_tests
        
        # If regular patterns failed, try more aggressive parsing
        return self._parse_aggressive(text)
    
    def _is_tabular_format(self, text: str) -> bool:
        """
        Check if the text appears to be in a tabular format.
        
        Args:
            text: Extracted text
            
        Returns:
            Boolean indicating if the text appears to be in tabular format
        """
        lines = text.split('\n')
        
        # Check for table headers
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains table headers
            has_test_header = any(re.search(pattern, line_lower) for pattern in self.table_header_patterns)
            has_value_header = any(re.search(pattern, line_lower) for pattern in self.value_header_patterns)
            
            if has_test_header and has_value_header:
                return True
        
        # Look for consistent delimiters
        delimiter_counts = {
            '|': sum(line.count('|') for line in lines),
            '\t': sum(line.count('\t') for line in lines),
            ',': sum(line.count(',') for line in lines)
        }
        
        # If there are consistent delimiters, it might be tabular
        if any(count > len(lines) * 1.5 for count in delimiter_counts.values()):
            return True
        
        return False
    
    def _parse_tabular_format(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text that appears to be in a tabular format.
        
        Args:
            text: Extracted text
            
        Returns:
            List of dictionaries containing lab test information
        """
        lines = text.split('\n')
        lab_tests = []
        
        # Try to identify the header line
        header_line_idx = -1
        header_columns = {}
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this line contains table headers
            has_test_header = any(re.search(pattern, line_lower) for pattern in self.table_header_patterns)
            has_value_header = any(re.search(pattern, line_lower) for pattern in self.value_header_patterns)
            has_range_header = any(re.search(pattern, line_lower) for pattern in self.range_header_patterns)
            
            if has_test_header and has_value_header:
                header_line_idx = i
                
                # Determine potential delimiters
                potential_delimiters = ['|', '\t']
                if ',' in line and not (line.count(',') > 5):  # Avoid misidentifying normal text
                    potential_delimiters.append(',')
                
                # Try each delimiter
                for delimiter in potential_delimiters:
                    if delimiter in line:
                        columns = [col.strip().lower() for col in line.split(delimiter)]
                        
                        # Map column indices
                        for j, col in enumerate(columns):
                            if any(re.search(pattern, col) for pattern in self.table_header_patterns):
                                header_columns['test_name'] = j
                            elif any(re.search(pattern, col) for pattern in self.value_header_patterns):
                                header_columns['value'] = j
                            elif any(re.search(pattern, col) for pattern in self.range_header_patterns):
                                header_columns['range'] = j
                        
                        # If we found at least test name and value columns, process with this delimiter
                        if 'test_name' in header_columns and 'value' in header_columns:
                            # Process data rows
                            for data_line_idx in range(header_line_idx + 1, len(lines)):
                                data_line = lines[data_line_idx].strip()
                                if not data_line:
                                    continue
                                
                                # Split by the same delimiter
                                data_cols = [col.strip() for col in data_line.split(delimiter)]
                                
                                # Skip if not enough columns
                                if len(data_cols) <= max(header_columns.values()):
                                    continue
                                
                                # Extract data
                                test_name = data_cols[header_columns['test_name']]
                                value_str = data_cols[header_columns['value']]
                                
                                # Get reference range if available
                                reference_range = "N/A"
                                if 'range' in header_columns and header_columns['range'] < len(data_cols):
                                    reference_range = data_cols[header_columns['range']]
                                
                                # Only add if test name and value are not empty
                                if test_name and value_str:
                                    lab_test = format_lab_test(test_name, value_str, reference_range)
                                    lab_tests.append(lab_test)
                            
                            # If we found lab tests, return them
                            if lab_tests:
                                return lab_tests
        
        # If the above method failed, try to identify columns by position
        if header_line_idx == -1:
            # Try to guess column positions based on common formats
            test_name_col = 0
            value_col = 1
            range_col = 2
            
            for line_idx in range(len(lines)):
                line = lines[line_idx].strip()
                if not line:
                    continue
                
                # Try tab as delimiter
                columns = line.split('\t')
                if len(columns) >= 2:
                    # Skip potential header lines
                    if any(re.search(pattern, columns[0].lower()) for pattern in self.table_header_patterns):
                        continue
                    
                    test_name = columns[test_name_col].strip()
                    value_str = columns[value_col].strip() if value_col < len(columns) else ""
                    
                    # Get reference range if available
                    reference_range = "N/A"
                    if range_col < len(columns):
                        reference_range = columns[range_col].strip()
                    
                    # Only add if test name and value are not empty
                    if test_name and value_str:
                        lab_test = format_lab_test(test_name, value_str, reference_range)
                        lab_tests.append(lab_test)
        
        return lab_tests
    
    def _parse_aggressive(self, text: str) -> List[Dict[str, Any]]:
        """
        Use more aggressive parsing techniques when standard patterns fail.
        
        Args:
            text: Extracted text
            
        Returns:
            List of dictionaries containing lab test information
        """
        lab_tests = []
        lines = text.split('\n')
        
        # Pattern for test names (common lab test names)
        test_name_patterns = [
            r"(Hemoglobin|WBC|RBC|Platelets|Glucose|Cholesterol|HDL|LDL|Triglycerides|Sodium|Potassium|Chloride|Calcium|Magnesium|Creatinine|BUN|ALT|AST|Bilirubin|Albumin|ALP|HbA1c)",
            r"(TSH|T3|T4|Vitamin D|B12|Folate|Iron|Ferritin)",
            r"([A-Za-z][A-Za-z\s\-]+)(?=\s*[:=])"
        ]
        
        # Pattern for numbers (potential test values)
        value_pattern = r"(\d+\.?\d*)"
        
        # Pattern for ranges
        range_pattern = r"(\d+\.?\d*\s*[-–]\s*\d+\.?\d*)"
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Try to find a test name
            test_name = None
            for pattern in test_name_patterns:
                match = re.search(pattern, line)
                if match:
                    test_name = match.group(1).strip()
                    break
            
            if not test_name:
                continue
                
            # Look for a value on this line or the next
            value_match = re.search(value_pattern, line[match.end():])
            value_str = None
            if value_match:
                value_str = value_match.group(1).strip()
            elif i + 1 < len(lines) and lines[i + 1].strip():
                value_match = re.search(value_pattern, lines[i + 1])
                if value_match:
                    value_str = value_match.group(1).strip()
            
            if not value_str:
                continue
                
            # Look for a reference range
            range_match = re.search(range_pattern, line)
            reference_range = "N/A"
            if range_match:
                reference_range = range_match.group(1).strip()
            elif i + 1 < len(lines) and lines[i + 1].strip():
                range_match = re.search(range_pattern, lines[i + 1])
                if range_match:
                    reference_range = range_match.group(1).strip()
            
            # Format the lab test
            lab_test = format_lab_test(test_name, value_str, reference_range)
            lab_tests.append(lab_test)
        
        return lab_tests

    def parse_table_data(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parse lab test information from table data.
        
        Args:
            table_data: 2D array of table cells
            
        Returns:
            List of dictionaries containing lab test information
        """
        lab_tests = []
        
        # Skip empty tables
        if not table_data or not table_data[0]:
            return lab_tests
        
        # Try to identify header row and columns
        header_row = 0
        test_name_col = -1
        value_col = -1
        range_col = -1
        
        # Check if first row looks like a header
        for j, cell in enumerate(table_data[0]):
            cell_lower = cell.lower()
            if any(re.search(pattern, cell_lower) for pattern in self.table_header_patterns):
                test_name_col = j
            elif any(re.search(pattern, cell_lower) for pattern in self.value_header_patterns):
                value_col = j
            elif any(re.search(pattern, cell_lower) for pattern in self.range_header_patterns):
                range_col = j
        
        # If we couldn't identify columns from header, make assumptions
        if test_name_col == -1 and value_col == -1:
            test_name_col = 0
            value_col = 1
            if len(table_data[0]) > 2:
                range_col = 2
            header_row = -1  # No header row
        
        # Process data rows
        for i in range(header_row + 1, len(table_data)):
            row = table_data[i]
            
            # Skip rows that don't have enough cells
            if len(row) <= max(col for col in [test_name_col, value_col, range_col] if col >= 0):
                continue
            
            # Extract data
            test_name = row[test_name_col].strip()
            value_str = row[value_col].strip()
            
            # Get reference range if available
            reference_range = "N/A"
            if range_col >= 0 and range_col < len(row):
                reference_range = row[range_col].strip()
            
            # Only add if test name and value are not empty
            if test_name and value_str:
                lab_test = format_lab_test(test_name, value_str, reference_range)
                lab_tests.append(lab_test)
        
        return lab_tests