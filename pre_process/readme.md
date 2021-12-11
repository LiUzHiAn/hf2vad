## Data preprocessing

### 0. Dataset preparing
Here we use the `ped2` dataset as example.

Download the [video anomaly detection dataset](http://101.32.75.151:8181/dataset/) and place it into 
the `data` directory of this project. In order to evaluate the frame-level AUC, we provide the 
frame labels of each test video in `data/ped2/ground_truth_demo/gt_label.json`. 

> For the Avenue and ShanghaiTech datasets, please check the last section in this page.   

The file structure should be similar as follows:
```python
./data
└── ped2
    ├── ground_truth_demo
    │   └── gt_label.json
    ├── testing
    │   └── frames
    │       ├── Test001
    │       ├── Test001_gt
    │       ├── Test002
    │       ├── Test002_gt
    │       ├── Test003
    │       ├── Test003_gt
    │       ├── Test004
    │       ├── Test004_gt
    │       ├── Test005
    │       ├── Test005_gt
    │       ├── Test006
    │       ├── Test006_gt
    │       ├── Test007
    │       ├── Test007_gt
    │       ├── Test008
    │       ├── Test008_gt
    │       ├── Test009
    │       ├── Test009_gt
    │       ├── Test010
    │       ├── Test010_gt
    │       ├── Test011
    │       ├── Test011_gt
    │       ├── Test012
    │       └── Test012_gt
    └── training
        └── frames
            ├── Train001
            ├── Train002
            ├── Train003
            ├── Train004
            ├── Train005
            ├── Train006
            ├── Train007
            ├── Train008
            ├── Train009
            ├── Train010
            ├── Train011
            ├── Train012
            ├── Train013
            ├── Train014
            ├── Train015
            └── Train016
```

### 1. Objects detecting
Please install the [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmaction2) accordingly, 
then download the cascade RCNN pretrained weights 
(we use `cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth`) and place it in `pre_porocess/assets` folder.

Run the following command to detect all the foreground objects. 
```python
$ python extract_bboxes.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract objects of all training data:
```python
$ python extract_bboxes.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=train
```
To extract objects of all test data:
```python
$ python extract_bboxes.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=test
```

After doing this, the results will be default saved at `./data/ped2/ped2_bboxes_train.npy`, 
in which each item contains all the bounding boxes in a single video frame.

### 2. Extracting optical flows
We extract optical flows in videos using use [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch). 

1. download the pre-trained FlowNet2 weights (i.e., `FlowNet2_checkpoint.pth.tar`) from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing) 
and place it in `pre_process/assets`.
2. build the customer layers via executing `install_custome_layers.sh`.
3. run the following command to estimate all the optical flows:
```python
$ python extract_flows.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract flows of all training data:
```python
$ python extract_flows.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=train
```
To extract flows of all test data:
```python
$ python extract_flows.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=test
```

After doing this, the estimated flows will be default saved at `./data/ped2/traning/flows`.
The final data structure should be similar as follows:
```python
./data
└── ped2
    ├── ground_truth_demo
    │   └── gt_label.json
    ├── ped2_bboxes_test.npy
    ├── ped2_bboxes_train.npy
    ├── testing
    │   ├── flows
    │   │   ├── Test001
    │   │   ├── Test002
    │   │   ├── Test003
    │   │   ├── Test004
    │   │   ├── Test005
    │   │   ├── Test006
    │   │   ├── Test007
    │   │   ├── Test008
    │   │   ├── Test009
    │   │   ├── Test010
    │   │   ├── Test011
    │   │   └── Test012
    │   └── frames
    │       ├── Test001
    │       ├── Test001_gt
    │       ├── Test002
    │       ├── Test002_gt
    │       ├── Test003
    │       ├── Test003_gt
    │       ├── Test004
    │       ├── Test004_gt
    │       ├── Test005
    │       ├── Test005_gt
    │       ├── Test006
    │       ├── Test006_gt
    │       ├── Test007
    │       ├── Test007_gt
    │       ├── Test008
    │       ├── Test008_gt
    │       ├── Test009
    │       ├── Test009_gt
    │       ├── Test010
    │       ├── Test010_gt
    │       ├── Test011
    │       ├── Test011_gt
    │       ├── Test012
    │       └── Test012_gt
    └── training
        ├── flows
        │   ├── Train001
        │   ├── Train002
        │   ├── Train003
        │   ├── Train004
        │   ├── Train005
        │   ├── Train006
        │   ├── Train007
        │   ├── Train008
        │   ├── Train009
        │   ├── Train010
        │   ├── Train011
        │   ├── Train012
        │   ├── Train013
        │   ├── Train014
        │   ├── Train015
        │   └── Train016
        └── frames
            ├── Train001
            ├── Train002
            ├── Train003
            ├── Train004
            ├── Train005
            ├── Train006
            ├── Train007
            ├── Train008
            ├── Train009
            ├── Train010
            ├── Train011
            ├── Train012
            ├── Train013
            ├── Train014
            ├── Train015
            └── Train016
```
### 3. Prefetch spatial-temporal cubes
For every extracted object above, we can construct a spatial-temporal cube (STC).
For example, assume we extract only one bbox in $i$-th frame, then we can crop the same region
from $(i-4), (i-3), (i-2), (i-1), i$ frames using the coordinates of that bbox, resulting a STC 
with shape `[5,3,H,W]`. Things are similar for the optical flows.

To extract all the STCs in the dataset, run the following command:
```python
$ python extract_samples.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract samples of all training data:
```python
$ python extract_samples.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=train
```
To extract samples of all test data:
```python
$ python extract_samples.py --proj_root=<path/to/project_root> --dataset_name=ped2 --mode=test
```
Note that the extracted samples number will be very large for Avenue and ShanghaiTech dataset,
hence we save the samples in a chunked file manner. The max number of samples in a separate
chunked file is set to be `100K` by default, feel free to modify that in [#Line11 here](./extract_samples.py)
depending on the available memory and disk space of your machine.

Given the first 4 frames and corresponding flows as input, the model is encouraged to predict the final frame.

___
After finishing the steps above, your dataset file structure should be similar as follows:
```python
./data
└── ped2
    ├── ground_truth_demo
    │   └── gt_label.json
    ├── ped2_bboxes_test.npy
    ├── ped2_bboxes_train.npy
    ├── testing
    │   ├── chunked_samples
    │   │   └── chunked_samples_00.pkl
    │   ├── flows
    │   │   ├── Test001
    │   │   ├── Test002
    │   │   ├── Test003
    │   │   ├── Test004
    │   │   ├── Test005
    │   │   ├── Test006
    │   │   ├── Test007
    │   │   ├── Test008
    │   │   ├── Test009
    │   │   ├── Test010
    │   │   ├── Test011
    │   │   └── Test012
    │   └── frames
    │       ├── Test001
    │       ├── Test001_gt
    │       ├── Test002
    │       ├── Test002_gt
    │       ├── Test003
    │       ├── Test003_gt
    │       ├── Test004
    │       ├── Test004_gt
    │       ├── Test005
    │       ├── Test005_gt
    │       ├── Test006
    │       ├── Test006_gt
    │       ├── Test007
    │       ├── Test007_gt
    │       ├── Test008
    │       ├── Test008_gt
    │       ├── Test009
    │       ├── Test009_gt
    │       ├── Test010
    │       ├── Test010_gt
    │       ├── Test011
    │       ├── Test011_gt
    │       ├── Test012
    │       └── Test012_gt
    └── training
        ├── chunked_samples
        │   └── chunked_samples_00.pkl
        ├── flows
        │   ├── Train001
        │   ├── Train002
        │   ├── Train003
        │   ├── Train004
        │   ├── Train005
        │   ├── Train006
        │   ├── Train007
        │   ├── Train008
        │   ├── Train009
        │   ├── Train010
        │   ├── Train011
        │   ├── Train012
        │   ├── Train013
        │   ├── Train014
        │   ├── Train015
        │   └── Train016
        └── frames
            ├── Train001
            ├── Train002
            ├── Train003
            ├── Train004
            ├── Train005
            ├── Train006
            ├── Train007
            ├── Train008
            ├── Train009
            ├── Train010
            ├── Train011
            ├── Train012
            ├── Train013
            ├── Train014
            ├── Train015
            └── Train016
```
The above steps also support Avenue and ShanghaiTech datasets. (May consume large disk space)

### Notice

- Avenue

For the [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) dataset, 
as explained in the dataset specifications, a few outliers are included in the training data. 
For instance, someone is running from 771-st frame to 831-st frame of training video 02, 
while running is treated as an abnormal event in test data. 
Considering that, we exclude frames contains such obvious anomalies from the training set. 
Specifically, they are 311-521(wrong direction) and 771-831(running) frames of training video 02,
1460-1510(wrong direction) frames of training video 04 and 741-900(wrong direction) frames of training video 07.

For simplicity, we cut the original training into two or three small videos if possible, obtaining 19 training videos in total(we call it Avenue19). 
One can download the Avenue19 [here](https://drive.google.com/file/d/1ygBf-Pbhbh1uWoBnGdirEcU6htfhMb-l/view?usp=sharing) directly. We strictly obey the original official ground-truth test labels in all our experiments.

> You still have to estimate the optical flows for yourself, since the disk space they cost makes it difficult to share.

- ShanghaiTech

For the [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) dataset, it only contains raw videos as the training data.
One can use ffmpeg tools to extract the frames, such as `ffmpeg -i <video_name> -qscale:v 1 -qmin 1 <video_name/%04d.jpg>`. The test frames are provided already.
