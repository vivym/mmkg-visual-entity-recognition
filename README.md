# 视觉实体识别使用说明

## 1. 算法描述
该算法用与图像的视觉实体识别。该算法基于PyTorch框架开发，输入一张图片，算法会识别图像中的实体的类别，输出类别名称及概率。

## 2. 环境依赖

CUDA版本: 11.7
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## 3. 下载安装

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-visual-entity-recognition
```

## 4. 使用示例及运行参数说明

```python
from PIL import Image
from mmkg_visual_entity_recognition import ImageEntityRecognition

image = Image.open("path/to/image")
label_id, label, prob = ImageEntityRecognition().inference(image)
```
