import cv2
import json
import threading
import time
import os

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from app.state import State
from app.constant import Mode, CropType, InputType, Interface, Device
from app.routine import routine

class Web:

  Frame_final = None
  
  def __init__(self) -> None:
    print(os.path.join(os.getcwd(), 'gui', 'dist'))
    self.app = Flask(__name__, 
                     template_folder=os.path.join(os.getcwd(), 'gui', 'dist'), 
                     static_folder=os.path.join(os.getcwd(), 'gui', 'dist', 'assets'), 
                     static_url_path='/assets/')
    self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # disable cache
    self.socketio = SocketIO(self.app, async_mode='threading')

    self.app.add_url_rule('/', '/', self.handle_index) 
    self.app.add_url_rule('/video_feed', '/video_feed', self.handle_video_feed) 
    # self.app.register_error_handler(404, self.page_not_found)

    self.socketio.on_event('run', self.handle_run)
    self.socketio.on_event('connect', self.handle_connect)
    self.socketio.on_event('stop', self.handle_stop)

  def init_setting(self, setting):
    tmp = json.dumps(setting)
    print(f'Send: {tmp}')
    self.socketio.emit('init', tmp)

  def handle_run(self, setting):
    msg = json.loads(setting)
    print(f'Receive: {msg}')
    State.setting = { 
      'mode': Mode(msg['mode']),
      'cropType': CropType(msg['cropType']),
      'saveFolder': msg['saveFolder'],
      'inputType': InputType(msg['inputType']),
      'inputPath': msg['inputPath'],
      'device': Device(msg['device']),
      'emulatorName': msg['emulatorName'],
      'model_path': msg['model_path'],
      'port': msg['port'],
      'grid': msg['grid'],
      'fps_limit': msg['fps_limit'],
      'bitrate': msg['bitrate'],
      'runtime': msg['runtime'],
      'interface': Interface(msg['interface']),
      'webport': msg['webport'],
      'delay': msg['delay'],
      'syncVideo': msg['syncVideo'],
      'yolov8': msg['yolov8']
    }
    def run_routine():
      routine()
      self.socketio.emit('finish')

    threading.Thread(target=run_routine, daemon=True).start()

  def handle_connect(self):
    self.init_setting(State.setting)

  def handle_stop(self):
    if State.running:
      State.stop = True

  def handle_index(self):
    return render_template('index.html')
  
  def gen_frames(self):
    while True:
      if State.Frame_final is None:
        time.sleep(0.2)
        continue
      else:
        _, buffer = cv2.imencode('.bmp', State.Frame_final)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/bmp\r\n\r\n' + frame + b'\r\n')
  
  def handle_video_feed(self):
    # return encoded frame to web
    return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
