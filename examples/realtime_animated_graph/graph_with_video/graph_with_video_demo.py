import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cv2

import os

output_dir = './output/'
video_file = 'Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4'
video_dir = '../../../../videos/'

cap = cv2.VideoCapture(os.path.join(video_dir, video_file))

# Read until video is completed
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
   
    # Display the resulting frame
    cv2.imshow('Frame', frame)
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break
   
# When everything done, release 
# the video capture object
cap.release()

df = pd.read_csv(os.path.join(output_dir, f'{video_file}.csv'))

on_list = df['on']
off_list = df['off']

x = []
time = 1

on_score = 0
off_score = 0

y_on = []
y_off = []

for frame in range(len(df)):

    on_score += on_list[frame]
    off_score += off_list[frame]

    if frame % 24 == 23:
        x.append(time)
        time += 1

        y_on.append(on_score / 24)
        y_off.append(off_score / 24)

        on_score = 0
        off_score = 0

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(x, y_off, label='off')
ax.plot(x, y_on, label='on')

ax.set_xticks(np.arange(0, x[-1], 5))

ax.set_xlabel('time (seconds)')
ax.set_ylabel('score')

ax.legend()

plt.show()

# Closes all the frames
cv2.destroyAllWindows()

