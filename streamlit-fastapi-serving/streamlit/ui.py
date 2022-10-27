# streamlit run your_script.py --server.port 30001
from genericpath import isfile
from urllib import response
import streamlit as st
import streamlit.components.v1 as components
import hydralit as hy
import hydralit_components as hc

import io, os, sys
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# TODO
# import asyncio
# import aiohttp
# import aiofiles


# from utils.reset import reset_data, reset_dir
# from utils.file_handler import make_image_from_video
# from utils.pose_filtering import pose_blur
# from utils.score2figure import make_figure_from_score
# from utils.make_blurred_video import encoding_video
# from utils.skip import skip
# from utils.mute import mute
# from utils.alternative import alternative

# from inference.kinetics_violence_localization import kinetics_violence_localization
# from inference.viodet_video import violence_detection

# sys.path.append("..")
# from blood_detection.yolov5.yolov5_custom.detect import main as blood_detection


# backend = "http://fastapi:8000/"
backend = "http://0.0.0.0:8000/"

# def process(image, server_url: str):
#     m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
#     r = requests.post(
#         server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
#     )
#     return r


def get_violence_video_call(api: str):
    rest_api = backend + api
    response = requests.get(rest_api)

    return response


def post_threshold_call(threshold: float, api: str):
    rest_api = backend + api
    data = {'threshold': threshold}
    response = requests.post(rest_api, json=data)

    return response


def post_violence_detection_call(video, api: str):
    rest_api = backend + api
    message = MultipartEncoder(fields={"video_file": (video.name, video, "video/mp4")})

    response = requests.post(
        rest_api, data=message, headers={"Content-Type": message.content_type}, timeout=300000
    )
    return response


def post_filtering_video_call(filter_num: int, api: str):
    # "violence_filtering"
    rest_api = backend + api
    data = {'filter_num': filter_num}
    response = requests.post(rest_api, json=data)

    return response


def get_skipped_video_call(api: str):
    # "violence_skipping"
    rest_api = backend + api
    response = requests.get(rest_api)

    return response


def get_muted_video_call(api: str):
    # "violence_muting"
    rest_api = backend + api
    response = requests.get(rest_api)

    return response


def get_alternative_video_call(api: str):
    # "violence_altering"
    rest_api = backend + api
    response = requests.get(rest_api)

    return response


def get_reset_data_call(api: str):
    rest_api = backend + api
    response = requests.get(rest_api)

    return response


app = hy.HydraApp(
  title='ZZOLFLIX',
  favicon="🎬",
  hide_streamlit_markers=True,
  allow_url_nav=True,
  navbar_sticky=False,
  #use_banner_images=[None,None,None,None,None],
  navbar_theme={'txc_inactive':'#D81F26', 'menu_background':'#FFFFFF', 'txc_active' : "#D81F26", 'option_active':'#FFFFFF'},
)


@app.addapp(title='', is_home=True, icon='')
def home():
    # CSS
    with open('assets/css/main.css') as f:
        CSS_TEXT = f.read()
    with open('assets/html/index.html') as f:
        HTML_TEXT = f.read()
    
        components.html(''
            f'<html>{HTML_TEXT}</html>'
            f'<style>{CSS_TEXT}</style>'
            , height=3800)

    hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                .css-18e3th9 {padding:0px;}
                .css-1gydwpt {display:block;}
                .css-1y3mfwq {display:block;}
                .css-1ln2a99 {display:block;}
                .css-8ralxy {display:block;}
                #complexnavbarSupportedContent>ul>li.active>a{color:red;}
                .navbar{display:none;}
            </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


@app.addapp(title='Get Started', icon='📽')
def test():
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            .css-18e3th9{padding:0px;}
            div.stButton > button:first-child {
                                                color: #D81F26;
                                                border-radius: 0.25rem;
                                                backgroud-color: #D81F26;
                                                height: 3rem;
                                                width: 30rem;
                                                box-sizing: 50%;
                                                margin: auto;
                                                font-size:18px;
                                                display: block;                      
                                                }
            .center {
                display: block;
                margin-left: auto;
                margin-right: auto;
            }              
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # path setting
    # xdviodet_path = 'project/XDVioDet'
    # images_path = 'data/images'
    # video_path = 'data/videos'
    # blurred_images_path = 'data/blurred_images'
    # audios_path = 'data/audios'
    # save_video_path = 'data/output_videos'

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        pass
    with col2:
        # st.title("Violence Detection")
        uploaded_file = st.file_uploader("Choose a Video", type=['mp4'])
        threshold = st.slider("Set a Threshold", min_value=0.00, max_value=1.0, step=0.05, value=0.8)

        # Violence Detection
        if st.button("Violence Detection"):
            if uploaded_file:
                # TODO async processing
                # asyncio.run(asyncio.wait(post_threshold_call(threshold, "set_threshold")))
                
                with hc.HyLoader('Violence Detection... Please Wait...', hc.Loaders.standard_loaders,index=5):
                    post_threshold_call(threshold, "set_threshold")
                    # violence detection
                    response_vd = post_violence_detection_call(uploaded_file, "violence_detection")
                    # download video
                    response_video = get_violence_video_call("violence_detection")
                
                st.video(response_video.content)
                # convert scored graph
                vd_score_image = Image.open(io.BytesIO(response_vd.content)).convert("RGB")
                st.image(vd_score_image, caption="Score_Output")
                vd_score_image.save("data/figures/score_output_1.png") # TODO if not exist dir -> create
            else:
                # handle case with no video
                st.write("Insert an video!")

        # Select Filter UI
        my_filter = ['blur', 'bubble']
        status = st.radio('Select the type of Filtering', my_filter)
        filter_num = 2 if status == my_filter[1] else 1
        # Violence Filtering
        if st.button("Violence Filtering"):
            vd_score_image = Image.open('data/figures/score_output_1.png')
            # TODO change check Violence Detection

            if vd_score_image:
                with hc.HyLoader('Violence Filtering... Please Wait...', hc.Loaders.standard_loaders,index=5):
                    post_threshold_call(threshold, "set_threshold")
                    # violence filtering and download video
                    response_video = post_filtering_video_call(filter_num, "violence_filtering")

                    st.video(response_video.content)
                    st.image(vd_score_image, caption="Score_Output")
            else:
                # handle case with no violence detection
                st.write("Start by Violence Detection!")

        # 영상 스킵
        if st.button("Skip Violent Scene"):
            with hc.HyLoader('Skip violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                post_threshold_call(threshold, "set_threshold")
                response_video = get_skipped_video_call("violence_skipping")
                st.video(response_video.content)

        # 영상 음소거
        if st.button("Mute Violent Scene"):
            with hc.HyLoader('Mute violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                post_threshold_call(threshold, "set_threshold")
                response_video = get_muted_video_call("violence_muting")
                st.video(response_video.content)

        # 대체 이미지
        if st.button("Alternative Violent Scene"):
            with hc.HyLoader('Alternative violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                post_threshold_call(threshold, "set_threshold")
                response_video = get_alternative_video_call("violence_altering")
                st.video(response_video.content)

        # 데이터를 초기화 하는 버튼
        if st.button("Reset All Data"):
            get_reset_data_call("reset_data")

    with col3:
        pass


if __name__ == "__main__":
    app.run()