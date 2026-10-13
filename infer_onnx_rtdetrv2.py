import onnxruntime as ort 
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import cv2
import copy
import numpy as np
import time


def infer_rtdetr(sess,image_path, th=0.4):
    original_im = Image.open(image_path).convert('RGB')
    original_size = original_im.size

    im = original_im.resize((640, 640))
    im_data = ToTensor()(im)[None]

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes":[list(original_size)]})

    labels, boxes, scores, queries = output

    return labels, boxes, scores, queries


