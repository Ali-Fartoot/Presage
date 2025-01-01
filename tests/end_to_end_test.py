# tests/test_end_to_end.py
import pytest
import requests
import os
from PIL import Image
import io

class TestEndToEnd:
    @pytest.fixture
    def base_url(self):
        """Fixture for base URL of the API"""
        return "http://localhost:8000"
    
    @pytest.fixture
    def test_image_path(self):
        """Fixture for test image path"""
        example_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'example')
        return os.path.join(example_dir, 'test.jpg')
    
    @pytest.fixture
    def test_image(self, test_image_path):
        """Fixture to create a test image if it doesn't exist"""
        if not os.path.exists(test_image_path):
            os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
            img = Image.new('RGB', (100, 100), color='white')
            img.save(test_image_path)
        return test_image_path

    def test_presage_endpoint(self, base_url, test_image):
        """Test the presage endpoint with an image file"""
        url = f"{base_url}/presage/"
        with open(test_image, "rb") as image_file:
            files = {
                "file": ("image.jpg", image_file, "image/jpeg")
            }
            response = requests.post(url, files=files)
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "filename" in response_data
        assert "analysis_result" in response_data
        assert isinstance(response_data["analysis_result"], str)

    def test_presage_endpoint_invalid_file(self, base_url):
        """Test the presage endpoint with invalid file type"""
        url = f"{base_url}/presage/"

        files = {
            "file": ("test.txt", io.StringIO("this is not an image"), "text/plain")
        }
        response = requests.post(url, files=files)
        assert response.status_code != 200
        response_data = response.json()
        assert "error" in response_data

    @pytest.mark.skip(reason="Server not running")
    def test_server_not_running(self, base_url):
        """Test behavior when server is not running"""
        url = f"{base_url}/presage/"
        
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.post(url, files={
                "file": ("test.jpg", io.BytesIO(b""), "image/jpeg")
            })