import requests
from pathlib import Path

# test end to end behavior
def test_image_upload(image):
    url = "http://localhost:8000/presage/"
    
    try:
        files = {"file": ("test.jpg", image, "image/jpeg")}
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

with open("./example/test.jpg", "rb") as img_file:
    result = test_image_upload(img_file)
    print(result)