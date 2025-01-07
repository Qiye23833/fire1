import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import tempfile

def load_model():
    """加载模型"""
    model_path = "./runs/train/expfire12/weights/best.pt"
    return YOLO(model_path)

def detect_fire(model, image):
    """进行火灾检测"""
    results = model.predict(
        source=image,
        conf=0.25,  # 置信度阈值
        line_width=2,  # 边框宽度
    )
    return results[0]  # 返回第一个结果

def main():
    st.title("火灾检测系统")
    st.write("上传图片进行火灾检测")

    # 加载模型
    @st.cache_resource  # 缓存模型
    def get_model():
        return load_model()
    
    model = get_model()

    # 文件上传
    uploaded_file = st.file_uploader("选择图片", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # 显示原始图片
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # 进行检测
        with st.spinner('正在检测...'):
            # 保存上传的文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # 检测
            results = detect_fire(model, tmp_path)
            
            # 获取检测结果
            boxes = results.boxes
            
            with col2:
                st.subheader("检测结果")
                # 显示处理后的图片
                plotted_img = results.plot()  # BGR to RGB
                st.image(plotted_img, use_column_width=True)
            
            # 显示检测详情
            st.subheader("检测详情")
            if len(boxes) > 0:
                for box in boxes:
                    # 获取类别和置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # 获取类别名称
                    class_name = model.names[cls]
                    st.write(f"检测到 {class_name}，置信度: {conf:.2f}")
            else:
                st.write("未检测到火灾")

            # 清理临时文件
            Path(tmp_path).unlink()

    # 添加说明
    with st.expander("使用说明"):
        st.write("""
        1. 点击"选择图片"上传要检测的图片
        2. 系统会自动进行火灾检测
        3. 检测结果会显示在右侧
        4. 检测详情会显示每个检测目标的类别和置信度
        """)

    with st.sidebar:
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25)
        st.write("当前置信度阈值:", conf_threshold)

if __name__ == "__main__":
    main() 