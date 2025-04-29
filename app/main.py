# Reintroducing comments to restore the original state
import os
import cv2
import numpy as np
import pandas as pd  # Add this import for handling the dataset
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
from glob import glob  # Add this import for handling file paths

from app.image_processor import preprocess_image, detect_table_regions, crop_to_roi, deskew_image
from app.text_extractor import TextExtractor
from app.lab_test_parser import LabTestParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lab Report Processor API",
    description="API for extracting lab tests from medical lab reports",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
text_extractor = TextExtractor()
lab_test_parser = LabTestParser()

@app.get("/")
async def root():
    """Root endpoint returning API health status."""
    return {"status": "healthy", "message": "Lab Report Processor API is running"}

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Process a lab report image and extract lab test information.
    
    Args:
        file: Uploaded lab report image file
    
    Returns:
        JSON response with extracted lab test information
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the image
        lab_tests = process_lab_report(image)
        
        # Return response
        return JSONResponse(content={"is_success": True, "lab_tests": lab_tests})
    
    except Exception as e:
        logger.error(f"Error processing lab report: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "is_success": False,
                "error": str(e)
            }
        )

def process_lab_report(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Process a lab report image and extract lab test information.
    
    Args:
        image: Lab report image as numpy array
    
    Returns:
        List of dictionaries containing lab test information
    """
    logger.info("Processing lab report image")
    
    # Deskew the image
    deskewed = deskew_image(image)
    
    # Preprocess the image
    processed = preprocess_image(deskewed)
    
    # Extract text from the entire image
    full_text = text_extractor.extract_text(processed)
    
    # Parse lab tests from the extracted text
    lab_tests = lab_test_parser.parse_text(full_text)
    
    # If we found tests, return them
    if lab_tests:
        logger.info(f"Extracted {len(lab_tests)} lab tests using full image OCR")
        return lab_tests
    
    # If no tests were found, try to detect table regions
    table_regions = detect_table_regions(deskewed)
    
    all_lab_tests = []
    
    # Process each table region
    for i, (x, y, w, h) in enumerate(table_regions):
        logger.info(f"Processing table region {i+1}/{len(table_regions)}")
        
        # Crop to the table region
        table_image = crop_to_roi(deskewed, x, y, w, h)
        
        # Preprocess the table image
        processed_table = preprocess_image(table_image)
        
        # Try to extract structured table data
        try:
            table_data = text_extractor.extract_table_data(processed_table)
            region_lab_tests = lab_test_parser.parse_table_data(table_data)
            all_lab_tests.extend(region_lab_tests)
        except Exception as e:
            logger.warning(f"Error extracting table data from region {i+1}: {str(e)}")
            
            # Fall back to regular text extraction
            region_text = text_extractor.extract_text(processed_table)
            region_lab_tests = lab_test_parser.parse_text(region_text)
            all_lab_tests.extend(region_lab_tests)
    
    # If we still haven't found any tests, try to extract structured data from the full image
    if not all_lab_tests:
        logger.info("Attempting structured data extraction from full image")
        try:
            structured_data = text_extractor.extract_structured_data(processed)
            
            # Construct text from structured data
            structured_text = "\n".join([line['text'] for line in structured_data])
            
            # Parse lab tests from the structured text
            all_lab_tests = lab_test_parser.parse_text(structured_text)
        except Exception as e:
            logger.warning(f"Error extracting structured data from full image: {str(e)}")
    
    # Remove duplicates based on test name
    unique_lab_tests = []
    test_names = set()
    
    for test in all_lab_tests:
        if test['test_name'] not in test_names:
            test_names.add(test['test_name'])
            unique_lab_tests.append(test)
    
    logger.info(f"Extracted {len(unique_lab_tests)} unique lab tests")
    return unique_lab_tests

@app.get("/dataset-info")
async def dataset_info():
    """
    Endpoint to retrieve basic information about the dataset.
    """
    return {"message": "Dataset information endpoint is under construction."}

# Define the dataset directory
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_reports_samples", "lbmaske")

# Get all PNG files in the dataset directory
def get_png_files():
    return glob(os.path.join(DATASET_DIR, "*.png"))

@app.get("/process-dataset")
async def process_dataset():
    """
    Endpoint to process all PNG files in the dataset directory.
    """
    png_files = get_png_files()
    if not png_files:
        raise HTTPException(status_code=404, detail="No PNG files found in the dataset directory")

    results = []
    for file_path in png_files:
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                logger.warning(f"Failed to read image: {file_path}")
                continue

            # Process the image
            lab_tests = process_lab_report(image)
            results.append({"file": os.path.basename(file_path), "lab_tests": lab_tests})
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    return {"processed_files": len(results), "results": results}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)