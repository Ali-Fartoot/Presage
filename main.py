from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from modules import FallbackAgent, PresageAgent, SegmentorAgent, HandLinesDetector

# pipeline
def pipeline(image):
    fallback = FallbackAgent()
    main_agent = PresageAgent()
    segment_model = SegmentorAgent()
    handline_detector = HandLinesDetector()

    # Check the given image consist of hand
    if "No" or "no" in fallback.infer(image=image):
        return "The given image doesn't consist of hand."

    segmented_image = segment_model(image)
    final_image = handline_detector(segmented_image)
    return main_agent.infer(final_image)

# Application
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
        result = pipeline(image)
        return {
            "filename": file.filename,
            "analysis_result": result
        }
        
    except Exception as e:
        return {"error": str(e)}