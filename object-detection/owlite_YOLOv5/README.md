# OwLite Object Detection Example 
- Model: YOLOv5-S
- Dataset: COCO'17 Dataset

## Prerequisites

### Prepare dataset
If you already have [COCO 2017 dataset](http://cocodataset.org), kindly modify the data path in within the "data/coco.yaml" file. Alternatively, executing the baseline will automatically download the dataset to the designated directory.

### Apply patch
```
cd YOLOv5
patch -p1 < ../apply_owlite.patch
```

### Setup environment
1. create conda env and activate it
    ```
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```
2. install required packages
    ```
    pip install -r requirements.txt
    ```
3. install OwLite package
    ```
    pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/SqueezeBits/owlite
    ```


## How To Run

### Run baseline model
```
python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name>
```

### Run quantized model
1. Create an experiment and save the config on OwLite GUI
2. Run the code for OwLite PTQ 
    ```
    python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --ptq
    ```
3. Run the code for OwLite QAT
    ```
    python train.py owlite --project <owlite_project_name> --baseline <owlite_baseline_name> --experiment <owlite_experiment_name> --qat
    ```

## Results

<details>
<summary>YOLOv5-S</summary>

### Configuration

#### Quantization Configuration

- Apply OwLite Recommended Config with the following calibration method
  - PTQ calibration: MSE
  - QAT backward: CLQ

#### Training Configuration

- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Epochs: 4
    
### Accuracy Results

| Quantization    | Input Size        | mAP 0.50~0.95 (%) | mAP 0.50 (%) |   
| --------------- |:-----------------:|:-----------------:|:------------:|
| FP32            | (32, 3, 640, 640) | 33.2              | 50.6         |
| OwLite INT8 PTQ | (32, 3, 640, 640) | 32.8              | 50.3         |
| OwLite INT8 QAT | (32, 3, 640, 640) | 33.4              | 51.7         |
| INT8 TensorRT   | (32, 3, 640, 640) | 28.7              | 45.2         |

- INT8 TensorRT engine was build using FP16 and INT8 flags, further explained in [TRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide)

### Latency Results
TensorRT Evaluation GPU: A6000

| Quantization    | Input Size        | GPU Latency (ms) | 
| --------------- |:-----------------:|:----------------:|
| FP16 TensorRT   | (32, 3, 640, 640) | 15.1             |
| OwLite INT8     | (32, 3, 640, 640) | 9.76             |
| INT8 TensorRT   | (32, 3, 640, 640) | 9.75             |

</details>

## Reference
https://github.com/ultralytics/yolov5