import cv2
import numpy as np
import math
import time
import logging

from typing import Tuple, List
from random import uniform

colors_rgb = []

def hex_to_rgb(hexa:str):
  """
    Convert hex value to rgb value.

    hexa: hex value.
  """

  return tuple(int(hexa[i:i+2], 16)  for i in (0, 2, 4))

def hsv_to_rgb(h, s, v):
  """
    Convert hsv value to rgb value.

    h: Hue.
    s: Saturation.
    v: Value.
  """
  if s == 0.0: v*=255; return (v, v, v)
  i = int(h*6.) # XXX assume int() truncates!
  f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
  if i == 0: return (v, t, p)
  if i == 1: return (q, v, p)
  if i == 2: return (p, v, t)
  if i == 3: return (p, q, v)
  if i == 4: return (t, p, v)
  if i == 5: return (v, p, q)

def getColors(k:int):
  """
    Get k different value.

    k: size of different color.
  """

  global colors_rgb
  while len(colors_rgb) < k:
    colors_rgb.append(hsv_to_rgb(uniform(0,1), uniform(0,0.4), uniform(0.75,1)))
  return colors_rgb

def drawCircles(img:cv2.Mat, circles:np.array, radius:int = 5, color:Tuple = (0,0,255), width:int = -1):
  """
    Draw circles on the image.

    img: input image, should be RGB image.
    circles: circles (x,y).
    radius: radius.
    color: color.
    width: thickness.
  """

  new_img = img.copy()
  for circle in circles:
    cv2.circle(new_img, circle, radius, color, width)

  return new_img

def drawLines(img:cv2.Mat, lines:np.array, color:Tuple = (0,0,255), width:int = 1):
  """
    Draw lines on the image.

    img: input image, should be RGB image.
    lines: lines (x0,y0,x1,y1).
    color: color.
    width: thickness.
  """

  new_img = img.copy()
  for line in lines:
    x0, y0, x1, y1 = line
    cv2.line(new_img, (x0, y0), (x1, y1), color, width)

  return new_img

def drawLinesClass(img:cv2.Mat, lines:np.array, labels:np.array, ignore:bool = True, width:int = 1):
  """
    Draw different class lines on the image.

    img: input image, should be RGB image.
    lines: lines.
    labels: classes of lines.
    ignore: whether ignore the lines with -1 class label.
    width: thickness.
  """

  new_img = img.copy()

  colors = getColors(max(labels)+2)
  for line, label in zip(lines, labels):
    if ignore and label == -1: continue
    x0, y0, x1, y1 = line
    color = colors[label+1]
    cv2.line(new_img, (x0, y0), (x1, y1), color, width)

  return new_img

def drawYoloLabel(img:cv2.Mat, locations:List, confidence:List, class_ids:List, class_names:List = None, width:int = 1):
  """
    Draw yolo detection result on img.

    img: input image.
    locations: list of box location.
    confidence: list of confidence.
    class_ids: list of class id.
    class_names: list of class name.
    width: thickness.
  """
  new_img = img.copy()  
  if len(class_ids) == 0: return new_img
  colors = getColors(len(class_names))

  for box, conf, cid in zip(locations, confidence, class_ids):
    left, top, w, h = box
    color = colors[cid]
    # draw box
    cv2.rectangle(new_img, (left, top), (left+w, top+h), color, width)
    # draw text
    label = f"{class_names[cid]}:{conf:.2f}"
    dim, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(new_img, (left, top), (left + dim[0], top + dim[1] + baseline), color, cv2.FILLED)
    cv2.putText(new_img, label, (left, top + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

  return new_img

def colLinePredictY(line, x): 
  """
    Given a X value and a line, calculate the Y value.

    line: line. y=ax+b, line[0]: slope (a), line[1]: intercept (b).
    x: X value.
  """

  return round(x*line[0]+line[1])

def lineLength(line):
  """
    Return length of line (EU).

    line: line. (x0,y0,x1,y1).
  """

  return math.sqrt(abs(line[2]-line[0])**2 + abs(line[3]-line[1])**2)

def cost_time(func):
  def wrap():
    start = time.time()
    r = func()
    end = time.time()
    print(f"[{func.__name__}] Cost {end - start} (s)")
    return r
  return wrap

def getLogger(name:str):
  """
    get logger.

    name: name of the logger.
  """

  logger = logging.getLogger(name=name)
  if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
  return logger