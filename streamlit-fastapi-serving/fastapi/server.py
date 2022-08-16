import io

from segmentation import get_segmentator, get_segments
from inference.viodet_video import get_violencer, get_violence
from starlette.responses import Response

from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel

# model = get_segmentator()
model_IFE, model_AFE = get_violencer()

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)

# @app.post("/set_threshold")
# def set_threshold(threshold: float):
#     print(threshold)


@app.post("/violence_detection")
def get_violence_score(video_file: UploadFile = File(...)):
    """Get violence score from video file"""
    file_location = f"data/videos/{video_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(video_file.file.read())
    print({"info": f"file '{video_file.filename}' saved at '{file_location}'"})

    violence_score_image = get_violence(model_IFE, model_AFE, 0.8)
    bytes_io = io.BytesIO()
    violence_score_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")