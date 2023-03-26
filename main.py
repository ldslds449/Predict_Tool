import json
import yaml
import os
import sys

from app.constant import Mode, CropType, InputType, Device, Interface
from argparse import ArgumentParser
from app.detect import RunTime
from app.state import State

parser = ArgumentParser()
parser.add_argument("--mode", help="tool mode", dest="mode", default="predict", type=Mode)
parser.add_argument("--cropType", help="crop type", dest="cropType", default="grid", type=CropType)
parser.add_argument("--save", help="save folder", dest="saveFolder", default="data", type=str)
parser.add_argument("--inputType", help="input type", dest="inputType", default="adb", type=InputType)
parser.add_argument("--input", help="input path", dest="inputPath", type=str)
parser.add_argument("--device", help="adb device", dest="device", default="bluestack", type=Device)
parser.add_argument("--emulatorName", help="emulator name", dest="emulatorName", type=str)
parser.add_argument("-M", "--model", help="ONNX Model", dest="model", default="best.onnx", type=str)
parser.add_argument("--grid", help="grid point (left-top,right-top,right-bottom,left-bottom)", dest="grid", default='{"540x960":"93,146,456,146,475,704,68,704", "720x1280":"118,196,596,196,630,934,83,934"}', type=str) # 540x960
parser.add_argument("-P", "--port", help="adb port", dest="port", default=5555, type=int)
parser.add_argument("--fps", help="FPS limit", dest="fps", default=60, type=int)
parser.add_argument("--bitrate", help="bitrate", dest="bitrate", default=16000000, type=int)
parser.add_argument("-C", "--config", help="config file path", dest="config", default="config.yaml", type=str)
parser.add_argument("--runtime", help="runtime", dest="runtime", default="onnx_runtime", type=RunTime)
parser.add_argument("--interface", help="interface", dest="interface", default="gui", type=Interface)
parser.add_argument("--webport", help="web port", dest="webport", default=8080, type=int)
parser.add_argument("--delay", help="screenshot delay", dest="delay", default=3000, type=int)
parser.add_argument("--syncVideo", help="sync the video to real play", dest="syncVideo", default=True, type=bool)
parser.add_argument("--yolov8", help="whether the model is yolov8", dest="yolov8", default=True, type=bool)

args = parser.parse_args()
config = args.config

# config file
if os.path.exists(config):
    with open(config, "r") as stream:
        data = yaml.safe_load(stream)
    mode = Mode(data["mode"])
    cropType = CropType(data["crop"]["type"])
    saveFolder = data["saveFolder"]
    delay = data["crop"]["delay"]
    model_path = data["model"]
    inputType = InputType(data["input"]["type"])
    inputPath = data["input"]["path"]
    device = Device(data["input"]["device"])
    emulatorName = data["input"]["emulatorName"]
    port = data["input"]["port"]
    fps_limit = data["input"]["fps"]
    bitrate = data["input"]["bitrate"]
    runtime = RunTime(data["runtime"])
    grid = data["grid"]
    interface = Interface(data["interface"]["type"])
    webport = data["interface"]["webport"]
    syncVideo = data["predict"]["syncVideo"]
    yolov8 = data["yolov8"]
else:
    # parameters
    mode = args.mode
    saveFolder = args.saveFolder
    cropType = args.cropType
    inputType = args.inputType
    inputPath = args.inputPath
    device = args.device
    emulatorName = args.emulatorName
    model_path = args.model
    port = args.port
    grid = json.loads(args.grid)
    fps_limit = args.fps
    bitrate = args.bitrate
    runtime = args.runtime
    interface = args.interface
    webport = args.webport
    delay = args.delay
    syncVideo = args.syncVideo
    yolov8 = args.yolov8

State.setting = { 
    'mode': mode,
    'saveFolder': saveFolder,
    'cropType': cropType,
    'inputType': inputType,
    'inputPath': inputPath,
    'device': device,
    'emulatorName': emulatorName,
    'model_path': model_path,
    'port': port,
    'grid': grid,
    'fps_limit': fps_limit,
    'bitrate': bitrate,
    'runtime': runtime,
    'interface': interface,
    'webport': webport,
    'delay': delay,
    'syncVideo': syncVideo,
    'yolov8': yolov8
}

print("====================================")
print(f"{'mode':18s}: {mode}")
print(f"{'saveFolder':18s}: {saveFolder}")
print(f"{'cropType':18s}: {cropType}")
print(f"{'inputType':18s}: {inputType}")
print(f"{'inputPath':18s}: {inputPath}")
print(f"{'device':18s}: {device}")
print(f"{'emulatorName':18s}: {emulatorName}")
print(f"{'model_path':18s}: {model_path}")
print(f"{'port':18s}: {port}")
print(f"{'grid':18s}: {grid}")
print(f"{'fps_limit':18s}: {fps_limit}")
print(f"{'bitrate':18s}: {bitrate}")
print(f"{'runtime':18s}: {runtime}")
print(f"{'interface':18s}: {interface}")
print(f"{'webport':18s}: {webport}")
print(f"{'delay':18s}: {delay}")
print(f"{'syncVideo':18s}: {syncVideo}")
print(f"{'yolov8':18s}: {yolov8}")
print("====================================")

if interface == Interface.GUI or interface == Interface.WEB:
    from app.web import Web
    import threading

    web = Web()
    run = lambda: web.socketio.run(web.app, port=webport, use_reloader=False)

    if interface == Interface.GUI:
        # start web server
        server = threading.Thread(target=run, daemon=True)
        server.start()
        # show the web
        import webview
        window = webview.create_window('Flask', f'http://127.0.0.1:{webport}/')
        webview.start()
    else:
        try:
            run()
        except:
            pass
else:
    from app.routine import routine
    routine()

print('Bye ~')
if State.running:
    State.stop = True
sys.exit(0)
    