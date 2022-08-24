import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference.viodet_video import get_violencer, get_violence
from inference.blood_detection import main as blood_detection
# from models.yolov5.detect import main as blood_detection

from inference.kinetics_violence_localization import kinetics_violence_localization
from starlette.responses import Response

from utils.file_handler import make_image_from_video
from utils.pose_filtering import pose_blur
from utils.score2figure import make_figure_from_score
from utils.make_blurred_video import encoding_video
from utils.skip import skip
from utils.mute import mute
from utils.alternative import alternative


threshold = 0.8
model_IFE, model_AFE = get_violencer()

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


class Item(BaseModel):
    threshold: float


@app.post("/set_threshold")
async def set_threshold(item: Item):
    threshold = item.threshold
    print({"info": f"set threshold '{threshold}'"})

    item_dict = item.dict()
    item_dict.update({"threshold": threshold})
    return item_dict


@app.post("/violence_detection")
async def get_violence_score(video_file: UploadFile = File(...)):
    """Get violence score from video file"""
    file_location = f"data/videos/{video_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(video_file.file.read())
    print({"info": f"file '{video_file.filename}' saved at '{file_location}'"})

    violence_score_image = get_violence(model_IFE, model_AFE, threshold)
    bytes_io = io.BytesIO()
    violence_score_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")


@app.get("/violence_detection")
async def get_violence_video():
    """Get violence video from video file"""
    some_file_path = f"data/output_videos/compatible_video.mp4"
    file_like = open(some_file_path, mode="rb")
    return StreamingResponse(file_like, media_type="video/mp4")


@app.get("/violence_filtering")
async def get_filtering_video():
    """Get violence video from video file"""
    # images_path = 'data/images'
    video_path = 'data/videos'
    blurred_images_path = 'data/blurred_images'
    audios_path = 'data/audios'
    save_video_path = 'data/output_videos'

    # blur_target_images_path = os.path.join(images_path, os.listdir(images_path)[0])
    # pose_blur(blur_target_images_path)
    make_image_from_video(video_path, blurred_images_path)
    blood_detection()
    kinetics_violence_localization(threshold)

    make_figure_from_score(
        threshold,
        off_path="data/npys/off.npy",
        on_path="data/npys/on.npy",
        index_path="data/list/output_index.list",
        save_path="data/figures",
    )
    encoding_video(blurred_images_path, audios_path, save_video_path)

    some_file_path = f"data/output_videos/encoding_video.mp4"
    file_like = open(some_file_path, mode="rb")
    return StreamingResponse(file_like, media_type="video/mp4")