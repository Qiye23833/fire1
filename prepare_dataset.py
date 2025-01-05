import os
import shutil
from pathlib import Path
import random

def prepare_dataset(source_dir, dest_dir, val_split=0.2):
    """
    整理数据集
    
    Args:
        source_dir: 源目录（包含 JPEGImages）
        dest_dir: 目标目录
        val_split: 验证集比例
    """
    # 创建必要的目录
    for split in ['train', 'val']:
        for type in ['images', 'labels']:
            os.makedirs(os.path.join(dest_dir, type, split), exist_ok=True)
    
    # 获取所有图片文件
    image_files = list(Path(source_dir).glob('*.jpg')) + \
                 list(Path(source_dir).glob('*.jpeg')) + \
                 list(Path(source_dir).glob('*.png'))
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 检查标签目录
    label_dir = Path(source_dir).parent / 'labels'
    if not label_dir.exists():
        print(f"警告：标签目录不存在: {label_dir}")
        return
    
    # 统计有效的图片（有对应标签的图片）
    valid_images = []
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_images.append((img_path, label_path))
    
    print(f"找到 {len(valid_images)} 个有效的图片-标签对")
    
    if not valid_images:
        print("错误：没有找到有效的图片-标签对！")
        return
    
    # 随机打乱
    random.shuffle(valid_images)
    
    # 计算验证集大小
    val_size = int(len(valid_images) * val_split)
    
    # 分割数据集
    train_pairs = valid_images[val_size:]
    val_pairs = valid_images[:val_size]
    
    print(f"训练集: {len(train_pairs)} 对")
    print(f"验证集: {len(val_pairs)} 对")
    
    # 移动文件
    for pairs, split in [(train_pairs, 'train'), (val_pairs, 'val')]:
        for img_path, label_path in pairs:
            # 移动图片
            shutil.copy2(
                img_path, 
                os.path.join(dest_dir, 'images', split, img_path.name)
            )
            
            # 移动标签
            shutil.copy2(
                label_path,
                os.path.join(dest_dir, 'labels', split, label_path.name)
            )
            
    print(f"数据集准备完成！")
    
    # 检查标签文件格式
    def check_label_format(label_path):
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:  # YOLO格式应该是5个数字
                        return False
                    class_id = int(parts[0])
                    if class_id != 0:  # 检查类别ID是否正确
                        print(f"警告：{label_path} 中存在非0类别ID: {class_id}")
                return True
        except:
            return False
    
    # 检查几个随机的标签文件
    sample_labels = random.sample([p[1] for p in valid_images], min(5, len(valid_images)))
    for label_path in sample_labels:
        if not check_label_format(label_path):
            print(f"警告：标签文件格式可能有问题: {label_path}")

if __name__ == '__main__':
    source_dir = r'D:\yolov11\datasets\fire\JPEGImages'  # 源目录
    dest_dir = r'D:\yolov11\datasets\fire'  # 目标目录
    
    prepare_dataset(source_dir, dest_dir) 