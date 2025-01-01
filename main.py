from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from modules import FallbackAgent, PresageAgent, SegmentorAgent, HandLinesDetector
import base64
import cv2
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

def convert_to_base64(image_bytes: bytes) -> str:
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"


def pipeline(image: bytes):
    base64_image = convert_to_base64(image)
    fallback = FallbackAgent()
    main_agent = PresageAgent()
    segment_model = SegmentorAgent()
    handline_detector = HandLinesDetector()
    fallback_result = fallback.infer(base64_image=base64_image)

    if any(x in fallback_result.lower() for x in ["no"]):
        return "The given image doesn't consist of hand."
    elif any(x in fallback_result.lower() for x in ["yes"]):
        segmented_image = segment_model(image)
        final_image = handline_detector(segmented_image)
        
        # Saving image
        cv2.imwrite('./example/final.jpeg', final_image)
        
        _, buffer = cv2.imencode(".png", final_image)
        image_bytes = buffer.tobytes()
        base64_image = convert_to_base64(image_bytes)
        return main_agent.infer(base64_image)
    else:
        return "Unrecognizable image!"

# App
app = FastAPI()
@app.post("/presage/")
async def analyze_image(file: UploadFile = File(...)):
    # Verify if file is an image
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"error": "File uploaded is not an image"}
        )
    
    try:
        # Read the image file
        content = await file.read()
        # Main alghorithm
        result = pipeline(content)
        
        return {
            "filename": file.filename,
            "analysis_result": result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )