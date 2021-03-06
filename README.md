# Hallucinated Hollow-3D R-CNN
<p align="center"> <img src='docs/framework.jpg' align="center" height="320px"> </p>

This is the official implementation of [**From Multi-View to Hollow-3D: Hallucinated Hollow-3D R-CNN for 3D Object Detection**](http://arxiv.org/abs/2107.14391), built on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). This paper has been accepted by IEEE TCSVT.

    @article{deng2021hh3d,
      title={From Multi-View to Hollow-3D: Hallucinated Hollow-3D R-CNN for 3D Object Detection},
      author={Deng, Jiajun and Zhou, Wengang and Zhang, Yanyong and Li, Houqiang},
      journal={arXiv:2107.14391},
      year={2021}
    }

### Installation
1.  Prepare for the running environment. 

    You can either use the docker image we provide, or follow the installation steps in [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). 

    ```
    docker pull djiajun1206/pcdet:pytorch1.6
    ```
2. Prepare for the data.

    Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):


    ```
    Voxel-R-CNN
    ├── data
    │   ├── kitti
    │   │   │── ImageSets
    │   │   │── training
    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
    │   │   │── testing
    │   │   │   ├──calib & velodyne & image_2
    ├── pcdet
    ├── tools
    ```
    Generate the data infos by running the following command:
    ```
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
    ```

3. Setup.

    ```
    python setup.py develop
    ```

### Getting Started
0. Downloading the model.

    The model reported in the manuscript can be download [here](https://drive.google.com/file/d/1NacQPEAUt7MTTRpfKhkqUQn-khO5Hb_M/view?usp=sharing).

1. Training.
    
    The configuration file is in tools/cfgs/kitti_models/hh3d_rcnn_car.yaml, and the training scripts is in tools/scripts.

    ```
    cd tools
    sh scripts/train_hh3d_rcnn.sh
    ```

2. Evaluation.

    The configuration file is in tools/cfgs/voxelrcnn, and the training scripts is in tools/scripts.

    ```
    cd tools
    sh scripts/eval_hh3d_rcnn.sh
    ```



### Acknowledge
Thanks to the strong and flexible [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) codebase.
