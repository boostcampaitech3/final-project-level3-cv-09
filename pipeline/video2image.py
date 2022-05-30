import os
import cv2
from utils  import createDirectory

def make_image_from_video(video_root_path, image_root_path):
    video_list = [os.path.join(video_root_path, i) for i in os.listdir(video_root_path)]
    image_list = [os.path.join(image_root_path, i) for i in os.listdir(video_root_path)]

    for file_path, save_path in zip(video_list, image_list):
        createDirectory(save_path[:-4])
        start = file_path.find('/', file_path.find('/') + 1)
        video_name = file_path[start+1:-4]
        print('convert mp4 to jpg :', video_name)

        cap = cv2.VideoCapture(file_path)
        FPS = cap.get(5) 
        frame = cap.get(7)
        count = 1

        if FPS == 30:
            for i in range(1, int(frame)+1):
                success, img = cap.read()
                
                if not success:
                    break

                if i % (1.25) < 1:
                    cv2.imwrite(file_path[:-4] + f"/{video_name}"+'-'+str(count).zfill(6)+'.jpg', img)
                    count+=1
        elif FPS == 24:
            for i in range(1, int(frame)+1):
                success, img = cap.read()

                if not success:
                    break

                if len(os.listdir(save_path[:-4])) == int(frame):
                    break

                height, width, channel = img.shape
                
                # if width > 640:
                #     d = width/640
                #     sc = 1/d
                #     img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)

                img = cv2.resize(img, (640, 360))

                cv2.imwrite(save_path[:-4] + f"/{video_name}"+'-'+str(count).zfill(6)+'.jpg', img)
                count+=1
        else:
            print("*"*30)
            print(f"FPS : {FPS} - {file_path}")
            assert FPS in [24, 30]
            print("*"*30)