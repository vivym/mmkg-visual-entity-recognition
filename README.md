# 视觉实体识别使用说明

## 1. 环境依赖

CUDA版本: 11.7
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 2. 下载安装

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-visual-entity-recognition
```

## 3. 使用示例及运行参数说明

```python
from PIL import Image
from mmkg_visual_entity_recognition import ImageEntityRecognition

image = Image.open("path/to/image")
label_id, label, prob = ImageEntityRecognition().inference(image)
```
