# streamlit run your_script.py --server.port 30001
from urllib import response
import streamlit as st
import streamlit.components.v1 as components
import hydralit as hy
import hydralit_components as hc

import json
import io, os, sys
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

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


def set_threshold_call(threshold, api):
    rest_api = backend + api

    data = {'threshold': threshold}
    response = requests.post(rest_api, data=json.dumps(data))
    return response


def violence_detection_call(video, api: str):
    rest_api = backend + api
    message = MultipartEncoder(fields={"video_file": (video.name, video, "video/mp4")})

    response = requests.post(
        rest_api, data=message, headers={"Content-Type": message.content_type}, timeout=300000
    )
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

        # # video save path
        # FILE_OUTPUT = 'data/videos'

        # if uploaded_file:
        #     with open(os.path.join(FILE_OUTPUT, uploaded_file.name), "wb") as out_file:  # open for [w]riting as [b]inary
        #         out_file.write(uploaded_file.read())

        threshold = st.slider("Set a Threshold", min_value=0.00, max_value=1.0, step=0.05, value=0.8)

        # Violence Detection
        if st.button("Violence Detection"):
            # with hc.HyLoader('Violence Detection... Please Wait...', hc.Loaders.standard_loaders,index=5):
            #     violence_detection(threshold)

            # image = Image.open('data/figures/score_output_1.png')
            # st.video(os.path.join(save_video_path, 'compatible_video.mp4'))
            # st.image(image, caption="Score_Output")
            if uploaded_file:
                # set_threshold_call(threshold, "set_threshold")
                response_vd = violence_detection_call(uploaded_file, "violence_detection")
                vd_score_image = Image.open(io.BytesIO(response_vd.content)).convert("RGB")
                st.image(vd_score_image, caption="Score_Output")
            else:
                # handle case with no video
                st.write("Insert an video!")

        # Violence Filtering
        if st.button("Violence Filtering"):
            # blur_target_images_path = os.path.join(images_path, os.listdir(images_path)[0])
            # with hc.HyLoader('Violence Filtering... Please Wait...', hc.Loaders.standard_loaders,index=5):
            #     # pose_blur(blur_target_images_path)
            #     make_image_from_video(video_path, blurred_images_path)
            #     blood_detection()
            #     kinetics_violence_localization(threshold)

            #     make_figure_from_score(
            #         threshold,
            #         off_path="data/npys/off.npy",
            #         on_path="data/npys/on.npy",
            #         index_path="data/list/output_index.list",
            #         save_path="data/figures",
            #     )

            # with hc.HyLoader('Save Video... Please Wait...', hc.Loaders.standard_loaders,index=5):
            #     encoding_video(blurred_images_path,
            #                     audios_path,
            #                     save_video_path)

            # st.video(os.path.join(save_video_path, 'encoding_video.mp4'))
            # image = Image.open('data/figures/score_output_1.png')
            # st.image(image, caption="Score_Output")
            pass
            
        # 영상 스킵
        if st.button("Skip Violent Scene"):
            with hc.HyLoader('Skip violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                # skip(threshold)
                pass

        # 영상 스킵
        if st.button("Mute Violent Scene"):
            with hc.HyLoader('Mute violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                # mute(threshold)
                pass

        # 대체 이미지
        if st.button("Alternative Violent Scene"):
            with hc.HyLoader('Alternative violent scenes... Please Wait...', hc.Loaders.standard_loaders,index=5):
                # alternative(threshold)
                pass

        # 데이터를 초기화 하는 버튼    
        if st.button("Reset All Data"):
            # reset_data()
            pass

    with col3:
        pass


if __name__ == "__main__":
    app.run()