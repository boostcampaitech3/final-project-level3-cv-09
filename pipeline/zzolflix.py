# streamlit run your_script.py --server.port 30001
import streamlit as st
import streamlit.components.v1 as components
import hydralit as hy
import hydralit_components as hc

import os
from PIL import Image

from reset import reset_data, setting_directory

from viodet_video import violence_detection

from pose_filtering import pose_blur
from make_blurred_video import encoding_video

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

count = 0

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
            </style>
            
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



    # path setting
    xdviodet_path = 'project/XDVioDet'
    images_path = 'data/images'
    audios_path = 'data/audios'
    save_video_path = 'data/output_videos'

    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.title("Violence Detection")

        uploaded_file = st.file_uploader("Choose a Video", type=['mp4'])

        # video save path
        FILE_OUTPUT = 'data/videos'

        if uploaded_file:
            with open(os.path.join(FILE_OUTPUT, uploaded_file.name), "wb") as out_file:  # open for [w]riting as [b]inary
                out_file.write(uploaded_file.read())

        # Violence Detection
        if st.button("Violence Detection"):
            with hc.HyLoader('Violence Detection... Please Wait...', hc.Loaders.standard_loaders,index=5):
                violence_detection()

            image = Image.open('data/figures/score_output_1.png')

            st.video(os.path.join(FILE_OUTPUT, os.listdir(FILE_OUTPUT)[0]))

            st.image(image, caption="Score_Output")

        # Violence Filtering
        if st.button("Violence Filtering"):
            blur_target_images_path = os.path.join(images_path, os.listdir(images_path)[0])
            with hc.HyLoader('Violence Filtering... Please Wait...', hc.Loaders.standard_loaders,index=5):
                pose_blur(blur_target_images_path)

            with hc.HyLoader('Save Video... Please Wait...', hc.Loaders.standard_loaders,index=5):
                encoding_video(images_path,
                                audios_path,
                                save_video_path)

            st.video(os.path.join(save_video_path, 'encoding_video.mp4'))
        
        # 데이터를 초기화 하는 버튼        
        if st.button("Reset All Data"):
            reset_data()

    with col3:
        pass

<<<<<<< HEAD
=======
        st.video(os.path.join(save_video_path, 'encoding_video.mp4'))
    
    # 데이터를 초기화 하는 버튼        
    if st.button("Reset All Data"):
        reset_data()
<<<<<<< HEAD
>>>>>>> 0b36b69e5945be8aa56f1ec5ef3c7820b4cb28a2
=======
>>>>>>> 0b36b69e5945be8aa56f1ec5ef3c7820b4cb28a2

app.run()
