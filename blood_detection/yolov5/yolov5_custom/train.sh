#!/bin/bash

# 실행은 터미널에서 
# bash {이 스크립트파일 이름} 

# ----- Custom Arg -----
# 아래 건들 필요 없이 여기부분만 커스텀하면 됩니다
# 따로 수정사항 없으면 주석처리!! ( ctrl+/ or cmd+/ )

weights="yolov5x6.pt" #(default : ROOT / 'yolov5s.pt')
data="blood.yaml" #(default=ROOT / 'data/coco128.yaml')
hyp="hyp.p6.yaml" #(default=ROOT / 'data/hyps/hyp.scratch.yaml')
epochs="300" #(default=300)
batch_size="16" #(default=16)
img_size="640" #(default=640)
workers="4" #(default=8)
project="../output/yolov5/" # project_dir (default=ROOT / 'runs/train')
name="movie_val20_10" # experiment_name (default='exp')
freeze="10" # Number of layers to freeze (default=0) backbone 10, all layer excluded final layer 24
# save_period="10" # Save checkpoint every x epochs (default=-1)

# ----------------------

cmd=""

function addCmd() {
	if [ -n "$2" ]; then
  	cmd+="--$@ "
	fi
}

addCmd "weights" ${weights:-""}
addCmd "data" ${data:-""}
addCmd "hyp" ${hyp:-""}
addCmd "epochs" ${epochs:-""}
addCmd "batch-size" ${batch_size:-""}
addCmd "imgsz" ${img_size:-""}
addCmd "workers" ${workers:-""}
addCmd "project" ${project:-""}
addCmd "name" ${name:-""}
addCmd "freeze" ${freeze:-""}
addCmd "save-period" ${save_period:-""}

# source /opt/conda/bin/activate;
# echo python3 -m train $cmd #혹시나 어떤 입력이 들어가는지 궁금하시다면
python3 -m train $cmd