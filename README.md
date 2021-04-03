# lesion_detection
Work on Universal Lesion Detection (ULD) on the NIH DeepLesion Dataset.

The following code is implemented on top of AlignShift(MICCAI 2020) and uses OpenMMLab's MMDetection as base framework. Thankyou to the authors for making their codebase available.
 

## Code structure

* ``alignshift``
  the core implementation of ACS convolutions, AlignShift convolution and TSM convolution, including the operators, models, and 2D-to-3D/AlignShift/TSM model converters. 
  * ``operators``: include ACS, AlignShiftConv, TSMConv.
  * ``converters.py``: include converters which convert 2D models to 3dConv/AlignShiftConv/TSMConv counterparts.
  * ``models``: Native ACS/AlignShift/TSM models. 
* ``deeplesion`` 
  the experiment code is base on [mmdetection](https://github.com/open-mmlab/mmdetection)
,this directory consists of compounents used in mmdetection.
* ``mmdet`` 


## How to run the experiments

* Dataset

  * Download [Deeplesion dataset](https://nihcc.box.com/v/DeepLesion)
  * Before training, mask should be generated from bounding box and recists. [mask generation](./deeplesion/dataset/generate_mask_with_grabcut.md)

* Preparing mmdetection script

  * Specify input ct slices in [./deeplesion/mconfigs/densenet_acs.py](./deeplesion/mconfigs/densenet_acs.py) through modifing NUM_SLICES in dict dataset_transform
  
  * Specify data root in [./deeplesion/ENVIRON.py](./deeplesion/ENVIRON.py)

* Training
  Using ```NCCL_DEBUG=INFO``` helps with getting the debug output causing the deadlock.
  ```bash
  ./deeplesion/train_dist.sh ${mmdetection script} ${dist training GPUS}
  ```
  * Train ACS models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfigs/densenet_acs.py 2
  ```
 * Evaluation 
   ```bash
   ./deeplesion/eval.sh ${mmdetection script} ${checkpoint path}
      ```
   ```bash
   ./deeplesion/eval.sh ./deeplesion/mconfigs/densenet_acs.py ./deeplesion/model_weights/acs_7slice.pth
   ```

**[WIP] More code is coming soon, stay tuned!**

