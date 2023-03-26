import time
import cv2
import os
import threading

from datetime import datetime
from app.detect import Detect
from app.state import State
from app.constant import *

def getBSPort(emulatorName):
    port_file = "C:\\ProgramData\\BlueStacks_nxt\\bluestacks.conf"
    with open(port_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not '=' in line: continue
            key, value = line.strip().split('=')
            if value == f'"{emulatorName}"':
                prefix = key.replace('display_name', 'status.adb_port')
        for line in lines:
            if not '=' in line: continue
            key, value = line.strip().split('=')
            if key == prefix:
                return int(value[1:-1])

def routine():
    try:
        State.running = True

        setting = State.setting
        mode = setting['mode']
        cropType = setting['cropType']
        saveFolder = setting['saveFolder']
        inputType = setting['inputType']
        inputPath = setting['inputPath']
        device = setting['device']
        emulatorName = setting['emulatorName']
        model_path = setting['model_path']
        port = setting['port']
        grid = setting['grid']
        fps_limit = setting['fps_limit']
        bitrate = setting['bitrate']
        runtime = setting['runtime']
        interface = setting['interface']
        delay = setting['delay']
        syncVideo = setting['syncVideo']
        yolov8 = setting['yolov8']

        # create folder
        if mode == Mode.CROP:
            img_folder = os.path.join(saveFolder, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            print(img_folder)
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        elif mode == Mode.RECORD:
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)

        if mode == Mode.PREDICT and interface == Interface.CLI:
            cv2.namedWindow('Predict', cv2.WINDOW_NORMAL)

        # load input
        d = None
        if inputType == InputType.ADB or mode == Mode.RECORD:
            from adbutils import adb
            if device == Device.BLUESTACK:
                try:
                    port = getBSPort(emulatorName)
                except:
                    pass
                print(f"BlueStack Port: {port}")
            try:
                adb.connect(f"127.0.0.1:{port}", timeout=3.0)
            except Exception as e:
                raise Exception(e)
            d = adb.device(f"127.0.0.1:{port}")
            print("Connected !")
            # window size
            screenSize = d.window_size()

        elif inputType == InputType.VIDEO:
            vcap = cv2.VideoCapture(inputPath)
            if not vcap.isOpened(): 
                raise Exception(f'Video is not found {inputPath}')
            video_fps = vcap.get(cv2.CAP_PROP_FPS)
            print("Finish Loading !")
            # window size
            screenSize = (int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        elif inputType == InputType.IMAGE:
            img = cv2.imread(inputPath, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception(f'Image is not found {inputPath}')
            print("Finish Loading !")
            # window size
            screenSize = img.shape[:2]

        # deal frame
        Frame = None
        client = None
        if inputType == InputType.ADB or mode == Mode.RECORD:
            import scrcpy
            def on_frame(frame):
                if frame is None: return
                nonlocal Frame
                Frame = frame

            # capture screenshot
            client = scrcpy.Client(device=d, max_fps=fps_limit, bitrate=bitrate, flip=False)
            client.add_listener(scrcpy.EVENT_FRAME, on_frame)
            client.start(threaded=True)

        # record
        if mode == Mode.RECORD:
            # codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # writer
            frameSize = (min(screenSize), max(screenSize))
            print(f"frameSize: {frameSize}")
            out = cv2.VideoWriter(os.path.join(saveFolder, f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.mp4'), 
                    fourcc, fps_limit, frameSize)
            print("Start Recording...")
            # grab frame
            while not State.stop:
                if Frame is None:
                    time.sleep(0.01)
                    continue
                out.write(cv2.resize(Frame, frameSize))
                State.Frame_final = Frame
                cv2.imshow('Record', Frame)
                cv2.setWindowTitle("Record", "Recording... (press 'q' to stop recording)")
                # check exit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    if client is not None: client.stop()
                    break
            cv2.destroyAllWindows()
            out.release()
            return

        # find the grid
        size_key = f"{min(screenSize)}x{max(screenSize)}"
        if size_key not in grid:
            raise Exception(f'Please check the window size {size_key}')
        grid_split = grid[size_key].split(',')
        grid_split = [int(x) for x in grid_split]
        aa, bb, cc, dd = (grid_split[0], grid_split[1]), (grid_split[2], grid_split[3]), (grid_split[4], grid_split[5]), (grid_split[6], grid_split[7])

        # initial detector
        detect = Detect()
        detect.initialModel(model_path, RTmode=runtime, yolov8=yolov8)

        # save screenshot
        def saveimgs(imgs):
            for img in imgs:
                name = f'screenshot_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}.png'
                cv2.imwrite(os.path.join(img_folder, name), img)
                print(name)

        Frame_bound = 20
        Frame_count = 0
        Frame_pre_time = None
        FPS = int(video_fps) if inputType == InputType.VIDEO else 0

        # main loop
        print("Start !!!")
        pre_adb_crop_time = None
        pre_read_time = None
        while not State.stop:
            # get frame
            if inputType == InputType.ADB:
                if Frame is None: continue
            elif inputType == InputType.VIDEO:
                pre_read_time = time.time()
                ret, Frame = vcap.read()
                if not ret:
                    if mode == Mode.PREDICT:
                        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    elif mode == Mode.CROP:
                        break
            elif inputType == InputType.IMAGE:
                if mode == Mode.CROP and Frame is not None: break
                Frame = img

            # do action
            if mode == Mode.PREDICT:
                # crop grid
                img_crop = detect.perspectiveTrans(Frame, aa, bb, cc, dd)

                # predict
                _, _, _, img_label = detect.findAlliedDians(img_crop, True)
                
                # show fps
                img_fps = cv2.putText(img_label, str(FPS), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

                # show predict result
                if interface != Interface.CLI:
                    State.Frame_final = img_fps
                else:
                    cv2.imshow('Predict', img_fps)
                    cv2.setWindowTitle("Predict", "Predict (press 'q' to close the client)")
                    # check exit
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        if client is not None: client.stop()
                        break
            
                end = time.time()

                # calculate FPS
                if Frame_pre_time is not None:
                    Frame_count += 1
                    if Frame_count % Frame_bound == 0:
                        Frame_time_sum = (end - Frame_pre_time)
                        FPS = round(Frame_bound / Frame_time_sum)
                        Frame_count = 0
                        Frame_pre_time = end
                else:
                    Frame_pre_time = time.time()

                if inputType == InputType.VIDEO:
                    if syncVideo:
                        time_per_frame = 1 / video_fps
                        delay_bias = 0.005
                        time.sleep(max(0, time_per_frame-(time.time()-pre_read_time) - delay_bias))
                

            elif mode == Mode.CROP:
                # crop grid
                img_crop = detect.perspectiveTrans(Frame, aa, bb, cc, dd)
                State.Frame_final = img_crop

                # save image
                if mode == Mode.CROP:
                    if inputType == InputType.ADB:
                        if pre_adb_crop_time is not None and time.time() - pre_adb_crop_time < delay / 1000: 
                            time.sleep(0.2)
                            continue
                        pre_adb_crop_time = time.time()

                    if cropType == CropType.GRID:
                        threading.Thread(target=saveimgs, args=[[img_crop]], daemon=True).start()
                    elif cropType == CropType.DIAN:
                        # predict
                        boxes, _, _ = detect.findAlliedDians(img_crop.copy())
                        # dian images
                        dians = detect.cropDians(img_crop, boxes)
                        threading.Thread(target=saveimgs, args=[dians], daemon=True).start()

                    # skip frame
                    if inputType == InputType.VIDEO:
                        skip_count = round((delay / 1000)*video_fps)
                        for _ in range(skip_count):
                            vcap.grab()


        if interface == Interface.CLI:
            cv2.destroyAllWindows()

        if client is not None:
            client.stop()

        if inputType == InputType.VIDEO:
            vcap.release()

        State.running = False
        State.stop = False
    except:
        State.running = False
        State.stop = False
