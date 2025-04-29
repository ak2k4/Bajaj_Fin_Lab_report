import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import io
from PIL import Image
import numpy as np

# Initialize test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_invalid_file_type():
    """Test uploading an invalid file type."""
    # Create a text file
    text_data = "This is not an image"
    response = client.post(
        "/get-lab-tests",
        files={"file": ("test.txt", text_data, "text/plain")}
    )
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]

def create_test_image(text="Hemoglobin: 14.2 (12.0-16.0)", size=(800, 600)):
    """Create a simple test image with text."""
    # Create a blank image
    image = Image.new('RGB', size, color='white')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_blank_image():
    """Test uploading a blank image."""
    img_bytes = create_test_image(text="", size=(100, 100))
    
    response = client.post(
        "/get-lab-tests",
        files={"file": ("blank.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    assert response.json()["is_success"] is True
    # Blank image should return empty lab tests
    assert len(response.json()["lab_tests"]) == 0

# Optional: Integration test with real image
# Uncomment and modify if you have test images available
"""
def test_real_lab_report():
    # Path to a real lab report image for testing
    test_image_path = "tests/data/sample_lab_report.jpg"
    
    if not os.path.exists(test_image_path):
        pytest.skip(f"Test image not found: {test_image_path}")
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/get-lab-tests",
            files={"file": ("sample_lab_report.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    assert response.json()["is_success"] is True
    assert len(response.json()["lab_tests"]) > 0
    
    # Check if at least one lab test has the expected structure
    lab_tests = response.json()["lab_tests"]
    assert "test_name" in lab_tests[0]
    assert "value" in lab_tests[0]
    assert "bio_reference_range" in lab_tests[0]
    assert "lab_test_out_of_range" in lab_tests[0]
"""