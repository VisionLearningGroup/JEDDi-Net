# Joint Event Detection and Description in Continuous Video Streams

Code released by Huijuan Xu (Boston University).

### Introduction

We present the Joint Event Detection and Description Network (JEDDi-Net) that solves the dense captioning task in an end-to-end fashion. Our model continuously encodes the input video stream with three-dimensional convolutional layers, proposes variable-length temporal events based on pooled features, and transcribes the event proposals into captions with the consideration of visual and language context.


### License

JEDDi-Net is released under the MIT License (refer to the LICENSE file for details).

### Citing JEDDi-Net

If you find JEDDi-Net useful in your research, please consider citing:

    @article{xu2019joint,
	title={Joint Event Detection and Description in Continuous Video Streams},
      	author={Xu, Huijuan and Li, Boyang and Ramanishka, Vasili and Sigal, Leonid and Saenko, Kate},
	journal={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
        year={2019}
    }


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Training](#training)
4. [Testing](#testing)

### Installation:

1. Clone the JEDDi-Net repository.
   ```Shell
   git clone --recursive git@github.com:VisionLearningGroup/JEDDi-Net.git
   ```

2. Build `Caffe3d` with `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)).

   **Note:** Caffe must be built with Python support!
  
  ```Shell
  cd ./caffe3d
  
  # If have all of the requirements installed and your Makefile.config in place, then simply do:
  make -j8 && make pycaffe
  ```

3. Build JEDDi-Net lib folder.

   ```Shell
   cd ./lib    
   make
   ```

### Preparation:

1. Download the ground truth annatations and videos in ActivityNet Captions dataset.

2. Extract frames from downloaded videos in 25 fps.

3. Generate the pickle data for training and testing JEDDi-Net model.

   ```Shell
   cd ./preprocess
   # generate training data
   python generate_train_roidb_sorted.py
   # generate validation data
   python generate_val_roidb.py  
   ```

### Training:
    
1. Download the separately-trained [segment proposal network(SPN)](https://drive.google.com/file/d/1GSaftfB1cprnKUOo8DSvbhp6SVAZbAMA/view) and [captioning](https://drive.google.com/file/d/1GbKl-0QnwIf1AvgFB-DQlbEoSJ3FVcQG/view) models ./pretrain/ .

2. In JEDDi-Net root folder, run:
   ```Shell
   bash ./experiments/denseCap_jeddiNet_end2end/script_train.sh
   ```

### Testing:

1. Download one sample JEDDi-Net model to ./snapshot/ .

   One JEDDi-Net model on ActivityNet Captions dataset is provided in: [caffemodel .](https://drive.google.com/file/d/1vtPeyPqqvsfNNfX16rUK7QsbrAEgxVxZ/view)

   The provided JEDDi-Net model has the METEOR score ~8.58% on the validation set.
   
   
2. In JEDDi-Net root folder, generate the prediction log file on the validation set.
   ```Shell
   bash ./experiments/denseCap_jeddiNet_end2end/test/script_test.sh 
   ```
   
3. Generate the results.json file from the prediction log file.
   ```Shell
   cd ./experiments/denseCap_jeddiNet_end2end/test/
   bash bash.sh
   ```
   
4. Follow the [evaluation code](https://github.com/ranjaykrishna/densevid_eval) to get the evaluation results.



