# 玉山無人飛行載具計數器比賽

## 比賽資訊
本競賽為空拍無人機偵測地面小物體之競賽，共有 4 個偵測類別（`car`, `hov`, `person`, `motocycle`），主辦方提供 1000 筆訓練資料，每張圖片的資料標籤為 txt 檔，txt 檔的每行分別代表圖片內物體對應的 `class`, `x`, `y`, `w`, `h`。

## EDA
1. **圖片尺寸分佈**: 881 張 `(1080, 1920)` 大小的圖片；119 張 `(720, 1344)` 大小的圖片。
2. 存在面積為 0 的物體，需進行過濾。

詳見 `notebooks/1-EDA.ipynb`

## 標籤轉換
將標籤轉換為 `yolov5`，詳見 `notebooks/2-convert_annotation.ipynb`。

## resize 圖片
圖片大小統一 resize 為 `(1080, 1920)` 大小，詳見 `notebooks/3-resize_image_to_1080_1920.ipynb`。

## 訓練、測試分割
對資料進行訓練、測試切分 `(0.85:0.15)`，詳見 `notebooks/4-train_test_split_yolov5_dataset.ipynb`。

## SAHI 切分圖片
使用 `SAHI` 套件對 **圖片** & **標籤** 進行切分，並將切分後的資料放入模型訓練，由於 `SAHI` 僅支援切分 **COCO** 格式標籤，因此，會切分 COCO 格式標籤後，再將 COCO 格式標籤轉換為 yolov5 格式標籤。
1. 切分 COCO 格式標籤詳見 `notebooks/6-sahi_coco_slice.ipynb`。
2. COCO 格式標籤轉換為 yolov5 格式標籤詳見 `notebooks/8-coco_to_yolov5.ipynb`。

最後，使用不同方式的切分方法進行訓練:

| name |  slice_height   | slice_width  | overlap_height_ratio | overlap_width_ratio |
| data1 |  ----  |  ----  |  ----  |  ----  |
| data2 | 1080  | 1100 | 0.25  | 0.25 |
| data3 | 600  | 752 | 0.20  | 0.20 |
| data4 | 624  | 600 | 0.25  | 0.25 |



## 模型訓練
1. **PPYOLOE**
    + **安裝 PaddleYOLO 套件**
        ```
        $ python -m pip install paddlepaddle-gpu==2.4.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
        $ pip install pycocotools
        $ git clone https://github.com/PaddlePaddle/PaddleYOLO  # clone
        $ cd PaddleYOLO
        $ pip install .
        ``` 
    + **下載預訓練模型權重檔**
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
                            -o LearningRate.base_lr=${base_lr} epoch=${epoch} \ 
                            worker_num=${worker_num} --eval 
        ```

2. **YOLOv5**
    + **安裝 YOLOv5 套件**
        ```
        $ git clone https://github.com/ultralytics/yolov5.git
        $ cd yolov5
        $ pip install -r requirements.txt
        ```

    + **下載預訓練模型權重檔**
    + **配置文件**
        ```yaml
        data/coco128.yaml

        # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
        # COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
        # Example usage: python train.py --data coco128.yaml
        # parent
        # ├── yolov5
        # └── datasets
        #     └── coco128  ← downloads here (7 MB)


        # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
        path: path/to/yolo/dataset/dir  # dataset root dir
        train: images/train  # train images (relative to 'path') 128 images
        val: images/val  # val images (relative to 'path') 128 images
        test:  # test images (optional)

        # Classes
        names:
        0 : car
        1 : hov
        2 : person
        3 : motorcycle   
        ```
    + **執行指令**
        ```
        $ python train.py --data coco128.yaml --epoch 150 --cfg hub/yolov5l6.yaml \
                        --weights ./weights/yolov5l6.pt --batch-size 6 --img 1280
        ```

3. **TPH-YOLOv5**
    + **安裝 TPH-YOLOv5 套件**
        ```
        $ git clone https://github.com/cv516Buaa/tph-yolov5.git
        $ cd tph-yolov5
        $ pip install -r requiremnts.txt
        $ pip install numpy==1.22.4
        $ pip install torch==1.11
        $ pip install torchvision==0.12
        ```

    + **下載預訓練模型權重檔**
    + **配置文件**
        ```yaml
        # data/coco128.yaml

        # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
        # COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
        # Example usage: python train.py --data coco128.yaml
        # parent
        # ├── yolov5
        # └── datasets
        #     └── coco128  ← downloads here


        # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
        path: path/to/yolo/dataset/dir  # dataset root dir
        train: images/train  # train images (relative to 'path') 128 images
        val: images/val  # val images (relative to 'path') 128 images
        test:  # test images (optional)

        # Classes
        nc: 4  # number of classes
        names: ['car', 'hov', 'person', 'motorcycle']  # class names
        ```

    + **執行指令**
      + **tph-yolov5l-xs-1**
        ```
        $ python train.py --img 1280 --adam --batch 4 --epochs 80 \
                        --data ./data/coco128.yaml --weights ./weights/yolov5l-xs-1.pt \ 
                        --hyp data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-xs-tph.yaml \
                        --name v5l-xs-tph
        ```
       
      + **tph-yolov5l-xs-2**
        ```
        $ python train.py --img 1280 --adam --batch 4 --epochs 120 \
                        --data ./data/coco128.yaml --weights ./weights/yolov5l-xs-2.pt \ 
                        --hyp data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-tph-plus.yaml \
                        --name v5l-tph-plus
        ```




