import cv2
import numpy as np
import os
import pytesseract as pt
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tets_for_fun import aws_textract
import csv

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# Import Load Yolo Model 
# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#function get detection
def get_detections(img,net):
  image = img.copy()
  row, col, d = image.shape
  max_rc = max(row,col)
  input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
  input_image[0:row,0:col] = image
  ##get prediction from yolo model
  blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
  net.setInput(blob)
  preds = net.forward()
  detections = preds[0]
  return input_image, detections


def non_max_supression(input_image,detections):
  boxes = []
  confidences = []
  image_w, image_h = input_image.shape[:2]
  x_factor = image_w/INPUT_WIDTH
  y_factor = image_h/INPUT_HEIGHT
    ####

  for i in range(len(detections)):
    row = detections[i]
    confidence = row[4] # confidence of detecting license plate
    if confidence > 0.4:
      class_score = row[5] # probability score of license plate
      if class_score > 0.25:
        cx, cy , w, h = row[0:4]
        left = int((cx - 0.5*w)*x_factor)
        top = int((cy-0.5*h)*y_factor)
        width = int(w*x_factor)
        height = int(h*y_factor)
        box = np.array([left,top,width,height])
        confidences.append(confidence)
        boxes.append(box)

  # clean
  boxes_np = np.array(boxes).tolist()
  confidences_np = np.array(confidences).tolist()
  # NMS
  index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
  return boxes_np, confidences_np, index


def drawings(image, boxes_np,confidences_np,index):  
    ##drawing 

  for ind in index:
    x,y,w,h =  boxes_np[ind]
    bb_conf = confidences_np[ind]
    conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
    license_text = extract_text(image,boxes_np[ind])



    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
    cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)
    cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
    cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
  return image



##Not comparing
numberis = []
numberis.append('up53aj9382')

def extract_text(image, bbox):
  output_file = open('./output/output.csv', mode='a', newline='')
  output_writer = csv.writer(output_file)
  x, y, w, h = bbox
  roi = image[y:y+h, x:x+w]
  if 0 in roi.shape:
    return ''
  else:
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
    text = str(aws_textract(magic_color))
    #text = text.strip()
    text = text.replace(' ', '')
    global numberis
    if len(text) >= 9 and (not numberis or text != numberis[-1]) and text not in numberis[-4:]:
    # if len(text) >= 9 and (not numberis or text != numberis[-1]):
      if len(numberis) > 0:
        print(text)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        object_filename = f"./object_detect/object_{timestamp}.jpg"
        cv2.imwrite(object_filename, image)
        plate_filename = f"./number_plate/plate_{timestamp}.jpg"
        cv2.imwrite(plate_filename, gray)
        output_writer.writerow([text, timestamp])
        
      numberis.append(text)
    numberis =  list(set(numberis))
    print(numberis)
    output_file.close()
    return text



def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
  if brightness != 0:
    if brightness > 0:
      shadow = brightness
      highlight = 255
    else:
      shadow = 0
      highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow
            
    buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
  else:
    buf = input_img.copy()
        
    if contrast != 0:
      f = 131*(contrast + 127)/(127*(131-contrast))
      alpha_c = f
      gamma_c = 127*(1-f)
            
      buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

  return buf



##combinin all the functions
def yolo_prediction(img,net):
    ##Step1 - prediction
  input_image, detections = get_detections(img,net)
    ##step2 - NMS
  boxes_np, confidences_np, index = non_max_supression(input_image,detections)
    ##Step3 - Drawings
  results_img = drawings(img, boxes_np,confidences_np,index)
  return results_img






##videos


cap = cv2.VideoCapture('./test_images/traffic.mp4')
while True:
  ret, frame = cap.read()
  if ret == False:
    print("unable to read video")

  results = yolo_prediction(frame,net)
  # cv2.namedWindow('results',cv2.WINDOW_KEEPRATIO)
  # cv2.imshow('YOLO',results)
  # if cv2.waitKey(1) == 27:
  #   break
cv2.destroyAllWindows()
cap.release()

        


    

