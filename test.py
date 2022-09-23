#!/usr/bin/env python3
import torchvision
import torch
import PIL
import numpy as np
from packaging import version
import platform
from pathlib import Path
from models.FaceMatch import FaceEmb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img1', type=Path)
    parser.add_argument('--path_img2', type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 1.1 # thresold to classify the imag to positive or negative (set to 1.1 same to facenet)
    transforms = torchvision.transforms.ToTensor()

    # Read the two images and transform them to tensors
    img1 = PIL.Image.open(args.path_img1).convert('RGB')
    img1 = transforms(img1)
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device)
    
    img2 = PIL.Image.open(args.path_img2).convert('RGB')
    img2 = transforms(img2)
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device)

    # define your model
    model = FaceEmb()
    model = model.to(device)
    model.load_state_dict(torch.load('./weights/model.pt'))

    # embedd the images into vectors
    out_img1 = model.backbone(img1)
    out_img2 = model.backbone(img2)
    
    # compute euclidian distance
    distance = torch.cdist(out_img1, out_img2)
    distance = distance.item()

    # decide whether it is the same person or it is different
    if distance <= threshold:
        print("Same, distance = ", distance)
    else:
        print("Different, distance =", distance)