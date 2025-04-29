# Reintroducing comments to restore the original state
import pytesseract
import cv2
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from pytesseract import Output

class TextExtractor:
    def __init__(self, lang='eng', config='--psm 6'):
        """
        Initialize the text extractor with Tesseract configurations.
        
        Args:
            lang: Language for OCR
            config: Tesseract configuration string
        """
        self.lang = lang
        self.config = config
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from the preprocessed image using Tesseract OCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Extracted text as string
        """
        text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
        return text
    
    def extract_text_with_boxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text with bounding box information.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of dictionaries containing text and bounding box information
        """
        # Get data from Tesseract
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )
        
        # Compile results
        results = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            # Skip empty text
            if int(data['conf'][i]) < 0 or not data['text'][i].strip():
                continue
                
            results.append({
                'text': data['text'][i],
                'confidence': data['conf'][i],
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
        
        return results
    
    def extract_structured_data(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract structured data from the image by analyzing text layout.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of structured text blocks with position information
        """
        # Get text with bounding boxes
        text_boxes = self.extract_text_with_boxes(image)
        
        # Group text by lines based on vertical position
        lines = self._group_by_lines(text_boxes)
        
        # Merge text within each line
        structured_lines = []
        for line in lines:
            line_text = " ".join([box['text'] for box in line])
            # Get the bounding box for the entire line
            x = min(box['x'] for box in line)
            y = min(box['y'] for box in line)
            width = max(box['x'] + box['width'] for box in line) - x
            height = max(box['y'] + box['height'] for box in line) - y
            
            structured_lines.append({
                'text': line_text,
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })
        
        return structured_lines
    
    def _group_by_lines(self, text_boxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group text boxes by lines based on their vertical position.
        
        Args:
            text_boxes: List of text boxes with position information
            
        Returns:
            List of lines, where each line is a list of text boxes
        """
        if not text_boxes:
            return []
        
        # Sort by y-coordinate
        sorted_boxes = sorted(text_boxes, key=lambda box: box['y'])
        
        lines = []
        current_line = [sorted_boxes[0]]
        
        # Tolerance for considering boxes on the same line
        line_height = sorted_boxes[0]['height']
        y_tolerance = line_height * 0.5
        
        for box in sorted_boxes[1:]:
            # Check if the box is on the same line as the current line
            current_y = current_line[0]['y']
            if abs(box['y'] - current_y) <= y_tolerance:
                current_line.append(box)
            else:
                # Sort boxes in current line by x-coordinate
                current_line = sorted(current_line, key=lambda b: b['x'])
                lines.append(current_line)
                current_line = [box]
                line_height = box['height']
                y_tolerance = line_height * 0.5
        
        # Add the last line
        if current_line:
            current_line = sorted(current_line, key=lambda b: b['x'])
            lines.append(current_line)
        
        return lines
    
    def extract_table_data(self, image: np.ndarray) -> List[List[str]]:
        """
        Extract data from a table structure in the image.
        
        Args:
            image: Preprocessed image containing a table
            
        Returns:
            2D array of table cells (rows and columns)
        """
        # Get structured lines
        lines = self.extract_structured_data(image)
        
        # Detect potential column boundaries based on text alignment
        columns = self._detect_columns(lines)
        
        # Organize text into rows and columns
        table_data = []
        current_row = [""] * len(columns)
        
        for line in lines:
            # Check which column this text belongs to
            assigned = False
            for i, (col_start, col_end) in enumerate(columns):
                if (line['x'] >= col_start - 10 and 
                    line['x'] < col_end + 10):
                    current_row[i] = line['text'].strip()
                    assigned = True
                    break
            
            # If we couldn't assign to a column, this might be a new row
            if not assigned and any(cell for cell in current_row):
                table_data.append(current_row)
                current_row = [""] * len(columns)
        
        # Add the last row if not empty
        if any(cell for cell in current_row):
            table_data.append(current_row)
        
        return table_data
    
    def _detect_columns(self, lines: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Detect potential column boundaries based on text alignment.
        
        Args:
            lines: List of structured lines
            
        Returns:
            List of column boundaries as (start_x, end_x) tuples
        """
        if not lines:
            return []
        
        # Collect all x-coordinates
        x_starts = [line['x'] for line in lines]
        x_ends = [line['x'] + line['width'] for line in lines]
        
        # Use histogram to find clusters of x-coordinates
        # This is a simplified approach; in a real system, you might use
        # more sophisticated clustering algorithms
        hist_bins = 20
        hist_range = (min(x_starts), max(x_ends))
        hist, bin_edges = np.histogram(x_starts, bins=hist_bins, range=hist_range)
        
        # Find peaks in the histogram
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 1:
                peak_indices.append(i)
        
        # Convert peak indices to x-coordinates
        column_starts = [bin_edges[i] for i in peak_indices]
        
        # Add the leftmost position as the first column start
        if not column_starts or column_starts[0] > min(x_starts) + 20:
            column_starts = [min(x_starts)] + column_starts
        
        # Create column boundaries
        columns = []
        for i in range(len(column_starts)):
            start = column_starts[i]
            if i < len(column_starts) - 1:
                end = column_starts[i+1] - 1
            else:
                end = max(x_ends)
            columns.append((start, end))
        
        return columns