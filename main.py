from tqdm import tqdm
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def convert_label(path, lb_path, xml_path, names):
    """将 XML 格式的标注转换为 YOLO 格式"""
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(lb_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text.lower()  # 转换为小写以确保匹配
            if cls in names:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)  # 获取类别ID
                out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')
            else:
                print(f"警告：未知类别 {cls} in {xml_path}")


def process_dataset(path, class_names):
    """处理数据集"""
    # 创建必要的目录
    for split in ['train', 'val','test']:
        (path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 处理训练集和验证集
    for split in ['train', 'val','test']:
        # 读取分割文件
        split_file = path / 'ImageSets' / f'{split}.txt'
        if not split_file.exists():
            print(f"错误：找不到{split}集分割文件：{split_file}")
            continue

        print(f"\n处理{split}集...")
        with open(split_file, 'r') as f:
            lines = f.readlines()

        # 处理每一行
        for line in tqdm(lines, desc=f'处理{split}集'):
            # 解析行内容
            parts = line.strip().split()
            if len(parts) == 2:  # 格式：images/xxx.jpg annotations/xxx.xml
                img_file = parts[0]
                xml_file = parts[1]
                
                # 从路径中提取文件名
                img_name = Path(img_file).name
                xml_name = Path(xml_file).name
                
                # 构建完整路径
                jpg_path = path / 'JPEGImages' / img_name
                xml_path = path / 'annotations' / xml_name
            else:
                print(f"警告：无效的行格式: {line}")
                continue
            
            if not jpg_path.exists():
                print(f"警告：找不到图片文件 {jpg_path}")
                continue
            if not xml_path.exists():
                print(f"警告：找不到标注文件 {xml_path}")
                continue

            # 目标路径
            dst_img = path / 'images' / split / img_name
            dst_label = path / 'labels' / split / f"{Path(img_name).stem}.txt"

            # 复制图片
            shutil.copyfile(jpg_path, dst_img)
            # 转换标注格式
            convert_label(path, dst_label, xml_path, class_names)


def check_dataset_structure(path):
    """检查数据集目录结构"""
    print("\n检查数据集结构...")
    print(f"根目录: {path}")
    
    # 更新必要的目录列表
    required_dirs = ['JPEGImages', 'annotations', 'ImageSets']
    required_files = ['ImageSets/train.txt', 'ImageSets/val.txt']
    
    # 检查必要的目录
    for dir_path in required_dirs:
        full_path = path / dir_path
        if full_path.exists():
            print(f"✓ 找到目录: {dir_path}")
            if dir_path == 'images':
                files = list(full_path.glob('*.jpg'))
                print(f"  - 包含 {len(files)} 个图片文件")
        else:
            print(f"✗ 缺少目录: {dir_path}")
    
    # 检查必要的文件
    for file_path in required_files:
        full_path = path / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                lines = f.readlines()
            print(f"✓ 找到文件: {file_path} ({len(lines)} 行)")
        else:
            print(f"✗ 缺少文件: {file_path}")


if __name__ == '__main__':
    # 设置路径和类别
    path = Path("D:/yolov11/datasets/fire2")
    class_names = ["fire", "smoke"]
    
    # 检查数据集结构
    check_dataset_structure(path)
    
    # 询问是否继续
    response = input("\n是否继续处理数据集？(y/n): ")
    if response.lower() != 'y':
        print("已取消处理")
        exit()
    
    # 处理数据集
    process_dataset(path, class_names)
    print("数据集处理完成！")
