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

## Model Zoo

<table style="width: 100%;">
    <tr>
        <th>No.</th>
        <th>Model</th>
        <th>Config</th>
        <th>Version</th>
        <th>Image Size</th>
        <th>Weight</th>
    </tr>
    <tr style="text-align:center">
        <td>1.</td>
        <td rowspan="4">YOLO v5</td>
        <td rowspan="4"><a href="https://github.com/ArkarPhyo1310/Vision-Kit/blob/develop/configs/yolov5.yaml">cfg</a></td>
        <td>s</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-D3Q7b-Ti4wcH_xAedjyeH0rAjsSWsEY/view?usp=share_link">weight</a></td>
    </tr>
    <tr style="text-align:center">
        <td>2.</td>
        <td>m</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-NWO_buw8vX3j7mUVxkAVuOX5ngB9pFG/view?usp=share_link">weight</a></td>
    </tr>
    <tr style="text-align:center">
        <td>3.</td>
        <td>l</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-MngB3003DmxLXBkoS0B830S0tVKDlaf/view?usp=share_link">weight</a></td>
    </tr>
    <tr style="text-align:center">
        <td>4.</td>
        <td>x</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-QTUuN-g9OkdS53MNlnK6kT8MMOVf7IU/view?usp=share_link">weight</a></td>
    </tr>
    <tr style="text-align:center">
        <td>5.</td>
        <td rowspan="2">YOLO v7</td>
        <td rowspan="4"><a href="configs/yolov7.yaml">cfg</a></td>
        <td>base</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-SupneyfNlaD1hmmOsOgBTH0Q4V1o3fM/view?usp=share_link">weight</a></td>
    </tr>
    <tr style="text-align:center">
        <td>6.</td>
        <td>x</td>
        <td>640</td>
        <td><a href="https://drive.google.com/file/d/1-V_RX4DUJjj2Iqwv4oqYyMHF7gfvuZKi/view?usp=share_link">weight</a></td>
    </tr>
</table>

### Supported Models

- YOLOv5
- YOLOv7

## Credits

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
