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
            cls = obj.find('name').text
            if cls in names:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)  # class id
                out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')
            else:
                print(f"警告：未知类别 {cls}")


def process_dataset(path, class_names):
    """处理数据集"""
    # 创建必要的目录
    for split in ['train']:
        (path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    jpg_files = list(Path(path / 'JPEGImages').glob('*.jpg'))
    print(f"\n找到 {len(jpg_files)} 个图片文件")

    # 处理所有图片
    for jpg_file in tqdm(jpg_files, desc='处理数据集'):
        # 获取对应的XML文件
        xml_file = path / 'Annotations' / f"{jpg_file.stem}.xml"
        
        if not xml_file.exists():
            print(f"警告：找不到标注文件 {xml_file}")
            continue

        # 目标路径
        dst_img = path / 'images/train' / jpg_file.name
        dst_label = path / 'labels/train' / f"{jpg_file.stem}.txt"

        # 复制图片
        shutil.copyfile(jpg_file, dst_img)
        # 转换标注格式
        convert_label(path, dst_label, xml_file, class_names)


def check_dataset_structure(path):
    """检查数据集目录结构"""
    print("\n检查数据集结构...")
    print(f"根目录: {path}")
    
    # 列出根目录下的所有目录
    print("\n现有目录:")
    for item in path.iterdir():
        if item.is_dir():
            print(f"- {item.name}/")
            # 列出子目录中的前5个文件
            try:
                files = list(item.iterdir())[:5]
                for f in files:
                    print(f"  - {f.name}")
            except Exception as e:
                print(f"  无法读取目录内容: {e}")


if __name__ == '__main__':
    # 设置路径和类别
    path = Path("D:/yolov11/datasets/fire")
    class_names = ["fire"]
    
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
