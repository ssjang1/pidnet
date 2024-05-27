import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
import models
import os

from ultralytics import YOLO

import argparse

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 설정 값들
# FILE_PATH = 'samples/result1.mp4'
# FILE_PATH = 'samples/test.png'
# FILE_NAME = os.path.basename(FILE_PATH)
# SAVE_DIR = 'output/'
FONTSCALE = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 2
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
color_map = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    # image -= mean
    # image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    logger.info('load_weights_start')
    logger.info(msg)
    logger.info('load_weights_done')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def find_contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# def is_person_near_road(pedestrian_mask, road_mask, threshold=1):
#     kernel = np.ones((3, 3), np.uint8)
#     dilated_road_mask = cv2.dilate(road_mask, kernel, iterations=threshold)
#     return np.any(np.logical_and(pedestrian_mask, dilated_road_mask))

def is_person_on_road(boxes, road_mask):
    for box in boxes:
        x1, y1, x2, y2 = map(int,box[:])
        if np.any(road_mask[y1:y2, x1:x2]):
            return True
    return False

def pad_image(image, stride=32):
    height, width = image.shape[:2]
    pad_height = (stride - height % stride) % stride
    pad_width = (stride - width % stride) % stride
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image


def process_image(file_path, model, yolov8_model):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    padded_img = pad_image(img)
    padded_height, padded_width, _ = padded_img.shape

    sv_img = np.zeros_like(padded_img).astype(np.uint8)
    img_transformed = input_transform(padded_img)
    img_transformed = img_transformed.transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).cuda()

    pred = model(img_tensor)
    pred = F.interpolate(pred, size=(padded_height, padded_width), mode='bilinear', align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    pred = pred[:height, :width]  # 패딩된 부분을 제거하여 원본 크기로 되돌림

    road_mask = (pred == 6).astype(np.uint8)
    
    yolov8_output = yolov8_model(img_tensor, classes=0, conf=0.3)[0]
    boxes = yolov8_output.boxes.xyxy.cpu().numpy()
    
    img_copy = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 바운딩 박스 그리기
        
        text_saram = 'person'
        (text_width, text_height), _ = cv2.getTextSize(text_saram, FONT, FONTSCALE, THICKNESS)
        text_x = x1
        text_y = y1 - 10  
        cv2.putText(img_copy, text_saram, (text_x, text_y), FONT, FONTSCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)

    person_on_road = is_person_on_road(boxes, road_mask)
    
    text = f'Person on road: {person_on_road}'
    COLOR = BLUE_COLOR if person_on_road else RED_COLOR

    (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONTSCALE, THICKNESS)
    x = (width - text_width) // 2
    y = 40 + text_height
    cv2.putText(img_copy, text, (x, y), FONT, FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    cv2.imwrite(os.path.join(SAVE_DIR, FILE_NAME), img_copy)
    
def process_video(file_path, model, yolov8_model):
    cap = cv2.VideoCapture(file_path)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(os.path.join(SAVE_DIR, FILE_NAME), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        padded_frame = pad_image(frame)
        padded_height, padded_width, _ = padded_frame.shape

        sv_img = np.zeros_like(padded_frame).astype(np.uint8)
        img = input_transform(padded_frame)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        pred = model(img)
        pred = F.interpolate(pred, size=(padded_height, padded_width), mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        pred = pred[:height, :width]  # 패딩된 부분을 제거하여 원본 크기로 되돌림

        road_mask = (pred == 6).astype(np.uint8)
        
        boxes = yolov8_model(img, classes=0, conf=0.3)[0].boxes.xyxy
        
        # pedestrian_mask = (pred == 5).astype(np.uint8)

        # for i, color in enumerate(color_map):
        #     if i == 6 :
        #         for j in range(3):
        #             sv_img[:height, :width, j][pred == i] = color_map[i][j]  # 패딩된 부분을 제외한 원본 크기 부분에만 색 적용

        # person_on_road = is_person_near_road(pedestrian_mask, road_mask)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int,box[:])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 바운딩 박스 그리기
            
            text_saram = 'person'
            (text_width, text_height), _ = cv2.getTextSize(text_saram, FONT, FONTSCALE, THICKNESS)
            text_x = x1
            text_y = y1 - 10  # 바운딩 박스 위쪽에 텍스트를 표시
            cv2.putText(frame, text_saram, (text_x, text_y), FONT, FONTSCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)

        person_on_road = is_person_on_road(boxes,road_mask)
        
        text = f'Person on road: {person_on_road}'
        COLOR = BLUE_COLOR if person_on_road else RED_COLOR

        (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONTSCALE, THICKNESS)
        x = (width - text_width) // 2
        y = 40 + text_height
        cv2.putText(frame, text, (x, y), FONT, FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame[:height, :width])  # 원본 크기로 다시 되돌림

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(description ='Custom Input')
    parser.add_argument('--f', help='Input image or video file path', default='samples/sample.png', type=str)
    parser.add_argument('--output', help='Directory path to save', default='output/', type=str)
    
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    
    is_cityscape = False
    model = models.pidnet.get_pred_model('pidnet-s', 19 if is_cityscape else 11)
    model = load_pretrained(model, 'weights/PIDNet_S_Camvid_Test.pt').cuda()
    model.eval()
    yolov8_model = YOLO('yolov8n.pt')
    FILE_PATH = args.f
    FILE_NAME = os.path.basename(FILE_PATH)
    SAVE_DIR = args.output

    ext = os.path.splitext(FILE_PATH)[-1].lower()
    if ext in ['.png', '.jpg', '.jpeg']:
        process_image(FILE_PATH, model, yolov8_model)
    elif ext in ['.mp4', '.avi', '.mov']:
        process_video(FILE_PATH, model, yolov8_model)
    else:
        raise ValueError("Unsupported file format")