## Multi-Scene Pose Regression with Transformers

This is a PyTorch implementation of multi-scene pose regression with Transformers described in our paper:
**Multi-Scene Pose Regression with Transformers**,   
 [[arXiv](https://arxiv.org/abs/????)]

---

### In a Nutshell

This code implements:

1. Training of a Transformer-based architecture for multi-scene pose regression 
2. Training of a PoseNet-like (CNN based) architecture for single scene pose regresion
3. Testing of the models implemented in 1-2

---

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Download the [Cambridge Landmarks(http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)] dataset and the [[7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)] dataset:

---

### Usage

The entry point for training and testing is the main.py script in the root directory


  For detailed explanation of the options run:
  ```
  python main.py -h
  ```
  For example, in order to train our model on the 7Scenes dataset run: 
  ```

  ```
  
  
  To run on cambridge, you will need to change the configuration of ems-transposenet in config.json 
  with the relevant configuration from the example_config.json file (e.g., the one under XXX) 
  
  In order to test a given checkpoint, provide the checkpoint path, for example:
  
  
  