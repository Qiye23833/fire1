from ultralytics.models import YOLO
import os
from pathlib import Path

def verify_dataset():
    """验证数据集完整性"""
    base_path = Path("./datasets/fire12")
    
    # 检查训练集
    train_imgs = list((base_path / 'images/train').glob('*.jpg'))
    train_labels = list((base_path / 'labels/train').glob('*.txt'))
    print(f"训练集图片数量: {len(train_imgs)}")
    print(f"训练集标签数量: {len(train_labels)}")
    
    # 检查验证集
    val_imgs = list((base_path / 'images/val').glob('*.jpg'))
    val_labels = list((base_path / 'labels/val').glob('*.txt'))
    print(f"验证集图片数量: {len(val_imgs)}")
    print(f"验证集标签数量: {len(val_labels)}")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 验证数据集
    verify_dataset()
    
    # 询问是否继续
    response = input("\n是否继续训练？(y/n): ")
    if response.lower() != 'y':
        print("已取消训练")
        exit()
    
    model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')
    model.train(data='./data.yaml', epochs=10, batch=16, device='0', imgsz=640, workers=4, cache=False,
                amp=True, mosaic=False, project='runs/train', name='exp')