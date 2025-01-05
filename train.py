from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')
    model.train(data='./data.yaml', epochs=20, batch=16, device='0', imgsz=640, workers=4, cache=False,
                amp=True, mosaic=False, project='runs/train', name='exp')