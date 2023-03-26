from enum import Enum

class Mode(str, Enum):
    CROP = 'crop'
    PREDICT = 'predict'
    RECORD = 'record'
class CropType(str, Enum):
    GRID = 'grid'
    DIAN = 'dian'
class InputType(str, Enum):
    ADB = 'adb'
    VIDEO = 'video'
    IMAGE = 'image'
class Device(str, Enum):
    BLUESTACK = 'bluestack'
    OTHER = 'other'
class Interface(str, Enum):
    CLI = 'cli'
    GUI = 'gui'
    WEB = 'web'