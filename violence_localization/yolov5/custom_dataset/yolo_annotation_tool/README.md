# Yolo-Annotation-Tool
### Get Started

1. 먼저 아래 링크에서 작업할 Data를 다운 받습니다.  
- [Download](https://drive.google.com/drive/folders/1RB7h9sSoaxnDV2nyXtCoty7jRC0sWLca?usp=sharing)
- 작업중인 데이터를 댓글에 작성해주세요 [Git-Issue](https://github.com/boostcampaitech3/final-project-level3-cv-09/issues/36)
</br>
</br>

2. 다운받은 압축파일을 하기 경로에 압축해제 해줍니다.
```
violence_localization/yolov5/custom_dataset/yolo_annotation_tool/Images/*.jpg
```
3. 실행 및 Annotation 시작
```
python main.py
```
4. 인터페이스
- b버튼으로 압축해제한 Images를 선택하면 load 됩니다.
- 사진상에서 시작꼭지점(클릭) -> 종료꼭지점(클릭)하면 라벨링 됩니다.
- 우상단의 annotation값을 클릭 후 delete를 누르면 삭제됩니다.
- 좌하단의 next, prev로 사진을 선택합니다.
- GUI 종료시 동일 이름의 .txt파일이 저장됩니다.
</br>
</br>

5. 종료 후 labeling 결과를 다시 drive에 압축하여 업로드
- ex) blood_0_complete.zip 업로드 [Link](https://drive.google.com/drive/folders/1RB7h9sSoaxnDV2nyXtCoty7jRC0sWLca?usp=sharing)