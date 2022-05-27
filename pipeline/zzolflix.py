# streamlit run your_script.py --server.port 30001
import streamlit as st
import streamlit.components.v1 as components
import hydralit as hy
import hydralit_components as hc

import os
from PIL import Image

from reset import reset_data

from viodet_video import violence_detection

from pose_filtering import pose_blur
from make_blurred_video import encoding_video

app = hy.HydraApp(title='ZZOLFLIX',
  favicon="🎬",
  hide_streamlit_markers=False,
  use_banner_images=[None,None,{'header':"<h3 style='text-align:center;padding: 0px 0px;color:black;font-size:200%; font-weight:800; color:#E50914;text-decoration:none'><a style='text-decoration:none'>ZZOLFLIX</a></h3><br>"},None,None],
  navbar_theme={'txc_inactive':'#000000', 'menu_background':'#FFFFFF', 'txc_active' : "#000000", 'option_active':'#FFFFFF'},
  )



@app.addapp(title='Home', is_home=True)
def home():
    # bootstrap
    components.html(
        """
        <html>

        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@500&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.14.0/css/all.css" integrity="sha384-HzLeBuhoNPvSl5KYnjx0BT+WB0QEEqLprO+NBkkk5gbc67FTaL7XIGa2w1L0Xbgc" crossorigin="anonymous">
        <style>
        .main_txt{font-size:40px; font-weight:bold;}
        .main_sm_txt{line-height: calc(1.4 + var(--space) / 100); font-size:22px; margin-top : 50px;}
        .f_box{display:inline-block; margin-right :50px; width:300px; padding: 50px 20px 0 50px; background-color : #ffffff; height:300px;}
        .main-tit {text-align: center;  margin-bottom: 80px; color: #1f3ec2; font-size: 14px;   text-transform: uppercase; letter-spacing: .2em;}
        .maintxt01{font-size: 40px;color: #111;font-weight: 200;}
        .f_box_txt{font-size:20px; margin-top:20px; margin-bottom:20px; color: #111; white-space: nowrap;}
        .f_box_txt_2{font-size: 16px; color: #414141; text-overflow: ellipsis; overflow: hidden; word-wrap: break-word; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;}

        .f_main_box{padding-left:40px;padding-top:70px; width:100%; height:600px; background-color : #f8f8f8; }
        .f_main_box_2{padding-left:100px;padding-top:170px; width:100%; height:400px; background-color :#ffffff; }

        .bluepoint{border-bottom : 3px solid #b4e7f8; box-shadow:inset 0 -4px #b4e7f8;}
        .maintxt02{font-size: 22px; color: #a6a6a6; margin-top: 40px;}
        .blue{ color: #2452c0;} 
        .footer-txt{color: #fff; font-size:18px;}
        .footer-txt2{font-size:25px; font-wight:bold; color: #fff;}

        .footer-box{border-top:1px solid #222; margin-top:100px; padding:100px 50px 100px 200px; background-color: #9ea1a9;}
        .f_copy{font-size:10px; color: #fff;}

        </style>
        <body style="font-family:Noto Sans KR">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-uWxY/CJNBR+1zjPWmfnSnVxwRheevXITnMqoEIeG1LJrdI0GlVs/9cVSyPYXdcSF" crossorigin="anonymous">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-kQtW33rZJAHjgefvhyyzcGF3C5TFyBQBA13V1RKPf4uH+bwyzQxZ6CmMZHmNBEfJ" crossorigin="anonymous"></script>
            <div id="carouselExampleCaptions" class="carousel slide" data-bs-ride="carousel">
                <div class="carousel-indicators">
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="1" aria-label="Slide 2"></button>
                <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="2" aria-label="Slide 3"></button>
            </div>
            <div class="carousel-inner">
                <div class="carousel-item active">
                <img src="https://www.pennmedicine.org/-/media/images/miscellaneous/fitness%20and%20sports/woman_exercise_home.ashx" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>Game with AI, Try it!</h5>
                    <p>Fun ways to get 30 minutes of physical activity today</p>
                </div>
                </div>
                <div class="carousel-item">
                <img src="https://image.urbanlifehk.com/wp-content/uploads/2021/10/087802b0.jpg" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>Let's have fun exercising together</h5>
                    <p>Anywhere, Anyone, with AI</p>
                </div>
                </div>
                <div class="carousel-item">
                <img src="https://www.tonicproducts.com/wp-content/uploads/2020/05/iStock-1141568835-scaled.jpg" class="d-block w-100" alt="...">
                <div class="carousel-caption d-none d-md-block">
                    <h5>How much do you exercise every day?</h5>
                    <p>It's exercise time</p>
                </div>
                </div>
            </div>
            
        </html>
        """,
        height=3680)

count = 0

@app.addapp(title='Test')
def test():

    # path setting
    xdviodet_path = '/opt/ml/input/code/project/XDVioDet'
    images_path = '/opt/ml/input/data/images'
    audios_path = '/opt/ml/input/data/audios'
    save_video_path = '/opt/ml/input/data/output_videos'


    st.title("Violence Detection")

    uploaded_file = st.file_uploader("Choose a Video", type=['mp4'])

    # video save path
    FILE_OUTPUT = '/opt/ml/input/data/videos'

    if uploaded_file:
        with open(os.path.join(FILE_OUTPUT, uploaded_file.name), "wb") as out_file:  # open for [w]riting as [b]inary
            out_file.write(uploaded_file.read())

    # Violence Detection
    if st.button("Violence Detection"):
        with hc.HyLoader('Violence Detection... Please Wait...', hc.Loaders.standard_loaders,index=5):
            violence_detection()

        image = Image.open(os.path.join(xdviodet_path, 'score_output_1.png'))

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
    # if st.button("Reset All Data"):
    #     reset_data()

app.run()
