# <div align="center"> VISION KIT </div>

## Pre-requisitics

Clone the repository and inside repository, run the following command.

```cmd
pip install -e .
```

## Training

```cmd
python scripts/main.py train --config <config/xxx.yaml>
```

## Evaluation

```cmd
python scripts/main.py eval --ckpt-path <checkpoint directory>
```

>NOTE: Checkpoint directory must contain "best.ckpt"

## Demo

```cmd
python scripts/demo.py --help
```

### Supported Models

- YOLOv5
- YOLOv7

## Credits

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
