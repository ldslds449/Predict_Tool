import cv2
import math
import pathlib
import os
import ctypes
import numpy as np

from typing import List
from enum import Enum
from dbscan1d.core import DBSCAN1D
from app.utils import *

import onnxruntime as rt

class RunTime(str, Enum):
  OPENCV_DNN = 'opencv_dnn'
  ONNXRUNTIME = 'onnx_runtime'
  OPENVINO = 'openvino'
  DEEPSPARSE = 'deepsparse'

class Backend(str, Enum):
  CPU = 'cpu'
  GPU = 'gpu'

class ExecuteMode(str, Enum):
  PARALLEL = 'parallel'
  SEQUENTIAL = 'sequential'

class Detect:
  logger:logging.Logger = None

  # model parameter
  RTmode:RunTime = None
  isYolov8:bool = None
  backend:Backend = None
  threadLimit:int = None
  executeMode:ExecuteMode = None
  EthreadLimit:int = None

  # model
  allied_model = None

  # input size
  allied_model_imgsz:Tuple = None

  # dll
  fastMethodDLL = None

  def __init__(self) -> None:
    self.logger = getLogger('detect')
    # load dll
    dll_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'libfastMethod.so')
    if os.path.exists(dll_path):
      try:
        self.fastMethodDLL = ctypes.PyDLL(dll_path)
        self.fastMethodDLL.filterBox.argtypes = [ctypes.py_object] * 9
        self.fastMethodDLL.filterBox.restype = ctypes.py_object
      except Exception as e:
        self.fastMethodDLL = None
        self.logger.info(f"Loading DLL error: {e}")
    else:
      self.logger.info(f"DLL not found, use pure python")

  def initialModel(self, allied_model_path:str, **kwargs):
    """
      Initial all predict model

      ---
      RTmode: Model RunTime
      yolov8: Whether the model is yolov8
      backend: Run model on cpu or gpu
      thread: Cpu thread limit
      Emode: Execute mode
      Ethread: Execute thread limit
    """
    
    # parameter
    def parseParameter(key, expectType, defaultValue):
      assert isinstance(defaultValue, expectType), f'Default value need to be {expectType} type'
      if key not in kwargs: return defaultValue
      return kwargs[key] if isinstance(kwargs[key], expectType) else defaultValue
    ## runtime
    self.RTmode = parseParameter('RTmode', RunTime, RunTime.ONNXRUNTIME)
    ## yolov8
    self.isYolov8 = parseParameter('yolov8', bool, True)
    ## backend
    self.backend = parseParameter('backend', Backend, Backend.CPU)
    ## thread limit
    self.threadLimit = parseParameter('thread', int, 1)
    ## execute Mode
    self.executeMode = parseParameter('Emode', ExecuteMode, ExecuteMode.SEQUENTIAL)
    ## thread limit 
    self.EthreadLimit = parseParameter('Ethread', int, 1)

    # load model
    def loadModel(mode_path):
      # model
      if self.RTmode == RunTime.ONNXRUNTIME: # onnx runtime
        sess_opt = rt.SessionOptions()
        sess_opt.intra_op_num_threads = self.threadLimit
        sess_opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL if self.executeMode == ExecuteMode.SEQUENTIAL else rt.ExecutionMode.ORT_PARALLEL
        sess_opt.inter_op_num_threads = self.EthreadLimit
        providers = ['CPUExecutionProvider'] if self.backend == Backend.CPU else ['CPUExecutionProvider', 'CUDAExecutionProvider']
        model = rt.InferenceSession(mode_path, sess_opt, providers=providers)
      elif self.RTmode == RunTime.OPENVINO: # openvino
        from openvino.runtime import Core
        ie = Core()
        model_onnx = ie.read_model(model=mode_path)
        model = ie.compile_model(model=model_onnx, device_name="CPU")
      elif self.RTmode == RunTime.OPENCV_DNN: # opencv dnn
        cv2.setNumThreads(self.threadLimit)
        model = cv2.dnn.readNet(mode_path)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU if self.backend == Backend.CPU else cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
      elif self.RTmode == RunTime.DEEPSPARSE: # deepsparse
        from deepsparse import Pipeline
        model = Pipeline.create(task=("yolov8" if self.isYolov8 else "yolo"), 
                                model_path=mode_path,
                                engine_type="deepsparse",
                                num_cores=self.threadLimit)
      # image size
      if self.RTmode == RunTime.ONNXRUNTIME:
        imgsz = tuple(model.get_inputs()[0].shape[2:])
      else:
        tmp = rt.InferenceSession(mode_path, providers=['CPUExecutionProvider'])
        imgsz = tuple(tmp.get_inputs()[0].shape[2:])
      self.logger.info(f"Model ({mode_path}) Input Shape: {imgsz}")
      return model, imgsz
    ## allied
    self.allied_model, self.allied_model_imgsz = loadModel(allied_model_path)

  def findLines(self, img:cv2.Mat, drawSeg = False):
    """
      Find all lines in the image.

      img: input image, should be gray scale.
      drawSeg: whether return the segment image.
    """
    
    lsd = cv2.createLineSegmentDetector(refine = cv2.LSD_REFINE_NONE, scale = 1.0, sigma_scale = 0.8, ang_th = 15)
    lines = lsd.detect(img)[0][:,0,:]

    lines_round = []
    for line in lines:
      lines_round.append([int(round(x)) for x in line])
    lines_round = np.array(lines_round)

    return lines_round if not drawSeg else (lines_round, drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lines_round))
  
  def findHorizontalLines(self, img:cv2.Mat, drawSeg:bool = False, tolerance:int = math.pi/60):
    """
      Find horizontal lines in the image.

      img: input image, should be gray scale.
      tolerate: theta tolerance (radian).
      drawSeg: whether return the segment image.
    """

    lines = self.findLines(img)
    hlines = []
    for line in lines:
      x0, y0, x1, y1 = line
      theta = math.atan2(y1-y0, x1-x0)
      if min(abs(theta - 0), abs(theta - math.pi)) <= tolerance:
        hlines.append(line)
    hlines = np.array(hlines)

    return hlines if not drawSeg else (hlines, drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), hlines))
  
  def findVerticalLines(self, img:cv2.Mat, drawSeg:bool = False, tolerance:int = math.pi/60):
    """
      Find vertical lines in the image.

      img: input image, should be gray scale.
      tolerate: theta tolerance (radian).
      drawSeg: whether return the segment image.
    """

    lines = self.findLines(img)
    vlines = []
    for line in lines:
      x0, y0, x1, y1 = line
      theta = math.atan2(y1-y0, x1-x0)
      if min(abs(theta - math.pi/2), abs(theta + math.pi/2)) <= tolerance:
        vlines.append(line)
    vlines = np.array(vlines)

    return vlines if not drawSeg else (vlines, drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), vlines))
  
  def findRows(self, img:cv2.Mat, drawSeg = False):
    """
      Find rows of the grid in the image.

      img: input image.
      hlines: all horizontal lines
      drawSeg: whether return the segment image.
    """

    H, W = img.shape[:2]
    hlines = self.findHorizontalLines(img)
    yMean = lambda x: (x[1]+x[3]) // 2

    # remove outlier
    rowlines = []
    for line in hlines:
      x0, y0, x1, y1 = line
      y = (y0 + y1) // 2
      length = abs(x1 - x0)
      if y < H*0.15 or y > H*0.8: continue
      if length < W*0.05: continue
      rowlines.append(line)

    # cluster lines
    yvalues = [yMean(x) for x in rowlines]
    dbs = DBSCAN1D(eps = 0.015*H, min_samples = 3)
    labels = dbs.fit_predict(np.array(yvalues))

    # remove center outlier
    yvalue_class = [[] for _ in range(max(labels)+1)]
    for line, label in zip(rowlines, labels):
      if label == -1: continue
      yvalue_class[label].append(line)

    centers = []
    for c in yvalue_class:
      longest = max(c, key = lambda x: abs(x[0] - x[2]))
      centers.append(yMean(longest))
    
    centers = sorted(centers, reverse=True)

    if drawSeg:
      cluster_img = drawLinesClass(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), rowlines, labels)
      row_img = drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), [[0, x, W, x] for x in centers])
      return (centers, cluster_img, row_img)
    else:
      return centers
    
  def findCols(self, img:cv2.Mat, drawSeg = False):
    """
      Find columns of the grid in the image.

      img: input image.
      drawSeg: whether return the segment image.
    """

    H, W = img.shape[:2]
    vlines = self.findVerticalLines(img)

    def xPitch(line):
      x0, y0, x1, y1 = line
      if x0 == x1:
        return line[0]
      else:
        slope = (y1-y0)/(x1-x0) if y1 < y0 else (y0-y1)/(x0-x1)
        return (H-y0)/slope + x0

    # remove outlier
    collines = []
    top_vlines = sorted(vlines, key=lambda x: lineLength(x), reverse=True)[:len(vlines)//3]
    for line in top_vlines:
      x0, y0, x1, y1 = line
      if min(y0, y1) < H*0.15 or max(y0, y1) > H*0.8: continue
      collines.append(line)

    # cluster lines
    xvalues = [xPitch(x) for x in collines]
    dbs = DBSCAN1D(eps = 0.025*W, min_samples = 3)
    labels = dbs.fit_predict(np.array(xvalues))

    # linear regression
    xvalue_class = [[] for _ in range(max(labels)+1)]
    for line, label in zip(collines, labels):
      xvalue_class[label].append(line)

    columns = []
    for c in xvalue_class:
      longest = max(c, key=lambda x: lineLength(x))
      x0, y0, x1, y1 = longest
      slope = (x1-x0)/(y1-y0) if y1 > y0 else (x0-x1)/(y0-y1)
      intercept = x0 - slope*y0
      columns.append((slope, intercept))

    columns = sorted(columns, key=lambda x: colLinePredictY(x, 0))

    if drawSeg:
      cluster_img = drawLinesClass(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), collines, labels)
      col_img = drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 
          [[round(colLinePredictY(x, 0)), 0, round(colLinePredictY(x, H)), H] for x in columns])
      return (columns, cluster_img, col_img)
    else:
      return columns 
    
  
  def findGrid(self, img:cv2.Mat, drawSeg = False, tolerate:Tuple[float, float] = (0.03, 0.035)):
    """
      Find the grid in the image and use Perspective Transform.

      img: input image, should be gray scale.
      drawSeg: whether return the segment image.
      tolerate: expand the grid, HW (percentage)
    """

    rows = self.findRows(img)
    cols = self.findCols(img)
    bottom, top = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    # calculate grid size
    grid_orig_height = lineLength([colLinePredictY(right, top), top, colLinePredictY(right, bottom), bottom])
    grid_orig_width = lineLength([colLinePredictY(right, bottom), bottom, colLinePredictY(left, bottom), bottom])
    
    # adjust grid with y tolerate
    trans_ratio = np.mean(np.diff(rows)) / (grid_orig_height/10)
    y_tol = round(tolerate[0]*grid_orig_height*trans_ratio)
    bottom = bottom - y_tol
    top = top + y_tol

    # adjust grid with x tolerate
    x_tol = round(tolerate[1]*grid_orig_width)
    left = (left[0], left[1] - x_tol)
    right = (right[0], right[1] + x_tol)

    a, b, c, d = [colLinePredictY(left, top), top], [colLinePredictY(right, top), top], [colLinePredictY(right, bottom), bottom], [colLinePredictY(left, bottom), bottom]

    if drawSeg:
      grid_img = drawLines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), [[*a, *b], [*b, *c], [*c, *d], [*d, *a]])
      return ((a, b, c, d), grid_img)
    else:
      return (a, b, c, d)
    

  def perspectiveTrans(self, img:cv2.Mat, a:List, b:List, c:List, d:List):
    """
      Apply Perspective Transform.

      img: input image, should be gray scale.
      a: left top.
      b: right top.
      c: right bottom.
      d: left bottom.
    """

    height = lineLength([b[0], b[1], c[0], c[1]])
    width = lineLength([c[0], c[1], d[0], d[1]])

    src_pts = np.array([a, b, c, d], dtype=np.float32)
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (round(width), round(height)))

    return warp
  
  def yoloPadding(self, img:cv2.Mat, new_shape:Tuple = (640, 640), padding_color:Tuple = (114, 114, 114)):
    """
      Apply Yolo padding.

      img: input image, should be color image.
      new_shape: output image shape (WH).
      padding_color: padding color.

      (Reference: https://medium.com/mlearning-ai/yolov5-avoid-these-common-mistakes-when-deploying-your-model-4567e86f6fde)
    """

    # resize while maintaining the aspect ratio
    H, W = img.shape[:2]

    # Scale ratio (new / old). 
    r = min(new_shape[0] / H, new_shape[1] / W)

    new_unpad = int(round(W * r)), int(round(H * r))

    # Resize the image
    img_resize = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # wh padding
    dw, dh = (new_shape[1] - new_unpad[0]), (new_shape[0] - new_unpad[1])  

    # divide padding into 2 sides
    dw /= 2  
    dh /= 2

    # compute padding on all corners
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_pad = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)  # add border

    return img_pad, (top, left), r
  
  def yoloPredict(self, model, img:cv2.Mat, drawSeg = False, class_names:List = None, new_shape:Tuple = (640, 640), conf_threshold:float = 0.5, class_threshold:float = 0.5, nms_threshold:float = 0.6):
    """
      Yolo predict routine.

      model: yolo model.
      img: input image, should be color image.
      drawSeg: whether return the segment image.
      new_shape: model input image shape.
      conf_threshold: threshold of confidence score.
      class_threshold: threshold of class score.
      nms_threshold: threshold of nms.
    """

    start = time.time()
    input_blob = None
    if self.RTmode == RunTime.DEEPSPARSE:
      # transform image
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      input_blob = img_rgb.transpose((2, 0, 1))

      self.logger.info(f"{'Transform':15s}: {time.time() - start:.8f} s")
      start = time.time()
    else:
      # padding
      img_pad, (orig_top, orig_left), scale_ratio = self.yoloPadding(img, new_shape=new_shape)

      self.logger.info(f"{'Padding':15s}: {time.time() - start:.8f} s")
      start = time.time()

      # transform image
      img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
      blob_img = np.float32(img_rgb) / 255.0
      blob_trans = blob_img.transpose((2, 0, 1))
      input_blob = np.expand_dims(blob_trans, 0)

      self.logger.info(f"{'Transform':15s}: {time.time() - start:.8f} s")
      start = time.time()
    
    # predict
    if self.RTmode == RunTime.ONNXRUNTIME:
      input_name = model.get_inputs()[0].name
      label_name = model.get_outputs()[0].name
      out = model.run([label_name], {input_name: input_blob})[0]
    elif self.RTmode == RunTime.OPENVINO:
      output_layer = model.output(0)
      out = model([input_blob])[output_layer]
    elif self.RTmode == RunTime.OPENCV_DNN:
      # set OpenCV DNN input
      model.setInput(input_blob)
      # OpenCV DNN inference
      out = model.forward()
    elif self.RTmode == RunTime.DEEPSPARSE:
      out = model(images=[input_blob])
    out = out[0]

    self.logger.info(f"{'Predict':15s}: {time.time() - start:.8f} s")
    start = time.time()

    # transpose
    if self.isYolov8:
      out = np.transpose(out)

      self.logger.info(f"{'Transpose':15s}: {time.time() - start:.8f} s")
      start = time.time()

    # find box
    if self.RTmode == RunTime.DEEPSPARSE:
      # class score threshold
      indices = np.flatnonzero(np.array(out.scores) >= class_threshold)
      boxes = np.array(out.boxes)[indices]
      if boxes.shape[0] > 0:
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]
        boxes = np.round(boxes).astype(np.int32)
      boxes = boxes.tolist()
      scores = np.array(out.scores)[indices].tolist()
      class_ids = np.array(out.labels)[indices]
      class_ids = [int(float(x)) for x in class_ids]
    else:
      if self.fastMethodDLL is not None:
        boxes, scores, class_ids = self.fastMethodDLL.filterBox( \
          out.tolist(), conf_threshold, class_threshold, orig_left, orig_top, scale_ratio, \
          img.shape[1], img.shape[0], self.isYolov8)
      else: # use pure python
        boxes, scores, class_ids = [], [], []
        for r in out:
          if self.isYolov8:
            centerX, centerY, w, h = r[:4]
            class_scores = r[4:]
          else:
            if r[4] < conf_threshold: continue # confidence threshold
            centerX, centerY, w, h, conf = r[:5]
            class_scores = r[5:]

          class_id = np.argmax(class_scores)
          class_score = class_scores[class_id]

          if class_score < class_threshold: continue # class score threshold

          # recover to padding img
          left = round(centerX - w/2) - orig_left
          top = round(centerY - h/2) - orig_top
          width = round(w)
          height = round(h)

          # recover to input img
          left = round(left / scale_ratio)
          top = round(top / scale_ratio)
          width = round(width / scale_ratio)
          height = round(height / scale_ratio)

          # adjust
          left = max(left, 0)
          top = max(top, 0)
          width = min(width, img.shape[1] - left)
          height = min(height, img.shape[0] - top)
          
          box = [left, top, width, height]
          boxes.append(box)
          scores.append(class_score if self.isYolov8 else conf)
          class_ids.append(class_id)

    self.logger.info(f"{'Filter Box':15s}: {time.time() - start:.8f} s")
    start = time.time()

    if self.RTmode != RunTime.DEEPSPARSE:
      # NMS
      boxes = np.array(boxes)
      scores = np.array(scores)
      indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)

      self.logger.info(f"{'NMS':15s}: {time.time() - start:.8f} s")
      start = time.time()

      # output
      class_ids = np.array(class_ids)
      boxes = [list(x) for x in list(boxes[indices])]
      scores = list(scores[indices])
      class_ids = list(class_ids[indices])

    if drawSeg:
      return boxes, scores, (class_ids, list(np.array(class_names)[class_ids])), drawYoloLabel(img, boxes, scores, class_ids, class_names)
    else:
      return boxes, scores, class_ids
  
  def findAlliedDians(self, img:cv2.Mat, drawSeg:bool = False):
    """
      Find allied dians with yolo.

      img: input image, should be color image.
      drawSeg: whether return the segment image.
    """

    assert self.allied_model is not None, "Allied Model Not Found"

    class_names = [
      "Assassin",
      "Barrier",
      "Battery",
      "Broken",
      "Charge",
      "Combat",
      "Death",
      "Engineer",
      "Fire",
      "Gargoyle",
      "Healing",
      "Ice",
      "Infect",
      "LevelUp",
      "Light",
      "Lightning",
      "Medusa",
      "Meteor",
      "Mimic",
      "Minigun",
      "Mushroom",
      "Nuclear",
      "Phoenix",
      "Poison",
      "Restoration",
      "Shield",
      "Spear",
      "Summon",
      "ThornShield",
      "Transcendence",
      "Vampire",
      "Wind",
      "Wolf",
      "Zombie",
    ]

    return self.yoloPredict(self.allied_model, img, drawSeg, class_names, new_shape=self.allied_model_imgsz)
  
  def blockMatching(self, img:cv2.Mat, boxes:List, drawSeg:bool = False):
    """
      Match dians to a block.

      img: input image, should be color image.
      boxes: all boxes location.
      drawSeg: whether return the segment image.
    """

    Y_RATIO = 0.1
    X_SIZE, Y_SIZE = 6, 5
    DIAN_RATIO = 1
    H, W = img.shape[:2]
    block_w, block_h = W // X_SIZE, H // (2*Y_SIZE)

    blocks = np.full((Y_SIZE, X_SIZE), -1, dtype=np.int32)
    anchors = []
    for i, box in enumerate(boxes):
      left, top, width, height = box

      # adjust those small dian
      width = max(width, block_w*DIAN_RATIO)
      height = max(height, block_h*DIAN_RATIO)

      anchor_x = round(left + (width / 2))
      anchor_y = round(height*(1-Y_RATIO) + top)
      if drawSeg: anchors.append([anchor_x, anchor_y])

      x_idx = anchor_x // block_w
      y_idx = Y_SIZE-1 - (H - anchor_y) // block_h

      blocks[y_idx][x_idx] = i;

    self.logger.info(f"Blocks: \n{blocks}")
    if drawSeg:
      lines = [[x*block_w, 0, x*block_w, H] for x in range(1, X_SIZE)]
      lines += [[0, H-y*block_h, W, H-y*block_h] for y in range(1, Y_SIZE*2)]
      img_block = drawLines(img, lines)
      img_block = drawCircles(img_block, anchors)
      return blocks, img_block
    else:
      return blocks

  def cropDians(self, img:cv2.Mat, boxes:List):
    """
      Crop all dians according to the predicted boxes.

      img: input image, should be color image.
      boxes: all boxes location.
    """

    images = []
    for box in boxes:
      left, top, width, height = box
      images.append(img[top:top+height, left:left+width,:])
    return images