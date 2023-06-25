# 玉山無人飛行載具計數器比賽

## 比賽資訊
本競賽為空拍無人機偵測地面小物體之競賽，共有 4 個偵測類別（`car`, `hov`, `person`, `motocycle`），主辦方提供 1000 筆訓練資料，每張圖片的資料標籤為 txt 檔，txt 檔的每行分別代表圖片內物體對聽的 `class`, `x`, `y`, `w`, `h`。

## 標籤轉換
將標籤轉換為 `yolov5` & `coco` 格式，標籤轉換詳見 `notebooks/convert_annotation.ipynb`。

## 圖片切分
使用 `SAHI` 套件對 **圖片** & **標籤** 進行切分，並將切分後的資料放入模型訓練，由於 `SAHI` 僅支援切分 **COCO** 格式標籤，因此，會切分 COCO 格式標籤後，再將 COCO 格式標籤轉換為 yolov5 格式標籤。
1. 切分 COCO 格式標籤詳見 `notebooks/slice_coco.ipynb`。
2. COCO 格式標籤轉換為 yolov5 格式標籤詳見 `notebooks/coco2yolov5.ipynb`。

最後，使用不同方式的切分方法進行訓練:

|  slice_height   | slice_width  | overlap_height_ratio | overlap_width_ratio |
|  ----  |  ----  |  ----  |  ----  |
| 1080  | 1100 | 0.25  | 0.25 |
| 624  | 600 | 0.25  | 0.25 |


## 模型訓練
1. **PaddleDetection**
    + **環境**
        ```
        $ python -m pip install paddlepaddle-gpu==2.4.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
        $ pip install pycocotools
        $ git clone https://github.com/PaddlePaddle/PaddleYOLO  # clone
        $ cd PaddleYOLO
        $ pip install .
        ``` 
    + **配置文件**:
        ```yaml
        # configs/smalldet/_base_/visdrone_sliced_640_025_detection.yml
        metric: COCO
        num_classes: 4

        TrainDataset:
            !COCODataSet
            : train2017
            anno_path: annotations/train2017.json
            dataset_dir: path/to/dataset/dir
            data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

        EvalDataset:
            !COCODataSet
            image_dir: val2017
            anno_path: annotations/val2017.json
            dataset_dir: path/to/dataset/dir
        TestDataset:
            !ImageFolder
            anno_path: annotations/val2017.json
            dataset_dir: path/to/dataset/dir
        ``` 

    + **執行指令**
        ```
        $ python tools/train.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml  \
                            -o LearningRate.base_lr=${base_lr} epoch=${epoch} 
                            worker_num=${worker_num} --eval 
        ```

2. **YOLOv5**

3. **TPH-YOLOv5**
