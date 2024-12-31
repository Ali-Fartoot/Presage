from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from modules import FallbackAgent, PresageAgent, SegmentorAgent, HandLinesDetector
import base64
# pipeline
def pipeline(base64_image: str):
    fallback = FallbackAgent()
    main_agent = PresageAgent()
    segment_model = SegmentorAgent()
    handline_detector = HandLinesDetector()
    print(fallback.infer(base64_image=base64_image))
    # Check the given image consist of hand
    if "No" or "no" in fallback.infer(base64_image=base64_image):
        return "The given image doesn't consist of hand."

    segmented_image = segment_model(base64_image)
    final_image = handline_detector(segmented_image)
    return main_agent.infer(final_image)


def convert_to_base64(image_bytes: bytes) -> str:
    """
    Convert image bytes to base64 data URI.
    
    Args:
        image_bytes (bytes): Image bytes
        
    Returns:
        str: Base64 encoded data URI
    """
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

app = FastAPI()
@app.post("/presage/")
async def analyze_image(file: UploadFile = File(...)):
    # Verify if file is an image
    if not file.content_type.startswith('image/'):
        return {"error": "File uploaded is not an image"}
    
    try:
        # Read the image file
        contents = await file.read()
        
        # Convert to base64
        base64_image = convert_to_base64(contents)
        image_req = f"data:image/png;base64,{base64_image}"
        # Create BytesIO object for PIL Image processing
        # image = Image.open(io.BytesIO(contents))
        
        # Your existing pipeline processing
        result = pipeline(image_req)
        
        return {
            "filename": file.filename,
            "base64_image": image_req,
            "analysis_result": result
        }
        
    except Exception as e:
        return {"error": str(e)}