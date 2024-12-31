import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import json
from main import app, pipeline

# Create test client
client = TestClient(app)

# Fixtures
@pytest.fixture
def test_image():
    """Create a test image"""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_agents(monkeypatch):
    """Mock all agents for testing"""
    class MockAgent:
        def infer(self, image):
            return "Mock response"
        
        def __call__(self, image):
            return image

    monkeypatch.setattr("main.FallbackAgent", lambda: MockAgent())
    monkeypatch.setattr("main.PresageAgent", lambda: MockAgent())
    monkeypatch.setattr("main.SegmentorAgent", lambda: MockAgent())
    monkeypatch.setattr("main.HandLinesDetector", lambda: MockAgent())

# Tests
def test_analyze_image_valid(test_image, mock_agents):
    """Test successful image analysis"""
    response = client.post(
        "/presage/",
        files={"file": ("test.png", test_image, "image/png")}
    )
    
    assert response.status_code == 200
    assert "filename" in response.json()
    assert "analysis_result" in response.json()
    assert response.json()["filename"] == "test.png"

def test_analyze_image_invalid_file():
    """Test invalid file type"""
    response = client.post(
        "/presage/",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 200  # Your endpoint returns 200 even for errors
    assert "error" in response.json()
    assert response.json()["error"] == "File uploaded is not an image"

def test_analyze_image_no_file():
    """Test missing file"""
    response = client.post("/presage/")
    assert response.status_code == 422  # FastAPI validation error

def test_pipeline_no_hand(test_image, monkeypatch):
    """Test pipeline when no hand is detected"""
    class MockFallbackAgent:
        def infer(self, image):
            return "No hand detected"
    
    monkeypatch.setattr("main.FallbackAgent", lambda: MockFallbackAgent())
    
    image = Image.open(test_image)
    result = pipeline(image)
    assert result == "The given image doesn't consist of hand."

def test_pipeline_with_hand(test_image, monkeypatch):
    """Test pipeline with hand detection"""
    class MockAgents:
        def infer(self, image):
            return "Hand detected"
        
        def __call__(self, image):
            return image
    
    monkeypatch.setattr("main.FallbackAgent", lambda: MockAgents())
    monkeypatch.setattr("main.PresageAgent", lambda: MockAgents())
    monkeypatch.setattr("main.SegmentorAgent", lambda: MockAgents())
    monkeypatch.setattr("main.HandLinesDetector", lambda: MockAgents())
    
    image = Image.open(test_image)
    result = pipeline(image)
    assert result == "Hand detected"

def test_analyze_image_exception(test_image, monkeypatch):
    """Test exception handling"""
    def mock_pipeline(image):
        raise Exception("Test error")
    
    monkeypatch.setattr("main.pipeline", mock_pipeline)
    
    response = client.post(
        "/presage/",
        files={"file": ("test.png", test_image, "image/png")}
    )
    
    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json()["error"] == "Test error"

# Integration tests
@pytest.mark.integration
def test_full_pipeline_integration(test_image):
    """Full integration test without mocks"""
    response = client.post(
        "/presage/",
        files={"file": ("test.png", test_image, "image/png")}
    )
    
    assert response.status_code == 200
    assert "filename" in response.json()
    assert "analysis_result" in response.json()

# Performance tests
@pytest.mark.performance
def test_response_time(test_image, mock_agents):
    """Test response time"""
    import time
    
    start_time = time.time()
    response = client.post(
        "/presage/",
        files={"file": ("test.png", test_image, "image/png")}
    )
    end_time = time.time()
    
    assert (end_time - start_time) < 2.0  # Response should be under 2 seconds