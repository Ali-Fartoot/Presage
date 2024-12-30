from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.post("/presage/")
async def analyze_image(file: UploadFile = File(...)):
    # Verify if file is an image
    if not file.content_type.startswith('image/'):
        return {"error": "File uploaded is not an image"}
    
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        

        return {
            "filename": file.filename,
            "analysis_result": None
        }
        
    except Exception as e:
        return {"error": str(e)}