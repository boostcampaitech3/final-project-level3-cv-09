import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2
import glob


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(
            os.path.join(image_dir, "images", vid, vid + "-" + str(i).zfill(6) + ".jpg")
        )[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226.0 - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        if w > 640:
            d = w / 640
            sc = 1 / d
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.0) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(
            os.path.join(image_dir, vid, vid + "-" + str(i).zfill(6) + "x.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )
        imgy = cv2.imread(
            os.path.join(image_dir, vid, vid + "-" + str(i).zfill(6) + "y.jpg"),
            cv2.IMREAD_GRAYSCALE,
        )

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224.0 - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.0) * 2 - 1
        imgy = (imgy / 255.0) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, "r") as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]["subset"] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == "flow":
            num_frames = num_frames // 2

        label = np.zeros((num_classes, num_frames), np.float32)

        fps = num_frames / data[vid]["duration"]
        for ann in data[vid]["actions"]:
            for fr in range(0, num_frames, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[ann[0], fr] = 1  # binary classification
        dataset.append((vid, label, data[vid]["duration"], num_frames))
        i += 1

    return dataset


def get_duration(video_start, video_end):
    dif_hour = int(video_end.split("-")[0]) - int(video_start.split("-")[0])
    dif_min = int(video_end.split("-")[1]) - int(video_start.split("-")[1])
    dif_sec = int(video_end.split("-")[2]) - int(video_start.split("-")[2])
    return dif_hour * 3600 + dif_min * 60 + dif_sec


def make_video_level_dataset(root, isTrain, num_classes=157):
    # split_file = split
    # split='testing'

    dataset = []

    files = glob.glob(root + "/videos" + "/*.mp4")
    # print(files)
    for file_path in files:
        file_name = file_path.split("/")[-1]
        if isTrain:
            violence_flag = file_name.split("label_")[-1]
            violence_flag = violence_flag.split(".")[0]

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(5)
        num_frames = int(cap.get(7) / fps * 24)
        duration = num_frames / 24
        # duration = get_duration(video_start, video_end)

        label = np.zeros((num_frames), np.float32)
        if isTrain:
            if violence_flag[0] == "B":  # ?????? ??????
                for fr in range(0, num_frames):
                    label[fr] = 1
            elif violence_flag[0] == "A":  # ????????? ??????
                for fr in range(0, num_frames):
                    label[fr] = 0
        dataset.append((file_name[:-4], label, duration, num_frames))

    return dataset


class Charades(data_utl.Dataset):
    def __init__(self, root, mode, isTrain, transforms=None, save_dir="", num=0):

        # video-level??? dataset ????????? ??? ?????? ??????
        self.data = make_video_level_dataset(root, isTrain)
        # self.data = make_dataset(split_file, split, root, mode)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid + ".npy")):
            return 0, 0, vid

        if self.mode == "rgb":
            imgs = load_rgb_frames(self.root, vid, 1, nf)
        else:
            imgs = load_flow_frames(self.root, vid, 1, nf)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
