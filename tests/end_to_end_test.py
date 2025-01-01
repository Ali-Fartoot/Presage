import requests

# URL of your FastAPI server
url = "http://localhost:8000/presage/"  # Adjust the URL if your server is running on a different host/port

# Path to your image file
image_path = "./example/test.jpg"  # Replace with your image path

# Open the image file
with open(image_path, "rb") as image_file:
    files = {
        "file": ("image.jpg", image_file, "image/jpeg")  
    }
    
    # Make the POST request
    response = requests.post(url, files=files)
    
    # Print the response
    if response.status_code == 200:
        print("Success!")
        print("Response:", response.json())
    else:
        print("Error:", response.status_code)
        print("Response:", response.text)