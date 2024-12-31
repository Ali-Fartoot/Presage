import requests
from pathlib import Path

# test end to end behavior
def test_image_upload(image_path):
    url = "http://localhost:8000/presage/"
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (Path(image_path).name, image_file, "image/jpeg")}
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Test the function
result = test_image_upload("./example/test.jpg")
print(result)