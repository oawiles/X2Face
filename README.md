This repository contains the code for [X2Face](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html), presented at ECCV 2018.

<h1>Demo Files for Trained Models</h1>

The demo notebooks demonstrate the following:
- How to load the pre-trained models
- How to drive a face with another face in `./UnwrapMosaic/Face2Face_UnwrapMosaic.ipynb`
- How to edit the embedded face with a drawing or tattoo in `./UnwrapMosaic/Face2Face_UnwrapMosaic.ipynb`
- How to drive with pose in `./UnwrapMosaic/Pose2Face.ipynb`
- How to drive with audio in `./UnwrapMosiac/Audio2Face.ipynb`

To run the notebooks, you need:
- pytorch=0.4.1 
- torchvision=0.2.1
- PIL
- numpy
- matplotlib

We tested the demo notebooks in this branch with pytorch 0.4.1, cuda 9.2 and python 2.7.
You can set up your environment like so:

```
conda install pytorch=0.4.1 cuda92 -c pytorch
pip install -r requirements.txt
```
The pretrained models might not work properly for other versions of pytorch.

Once the environment is set up, the pre-trained models can be downloaded from the [project page](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html) and the model paths in the notebooks updated appropriately (this should simply require setting the BASE_MODEL_PATH in the notebook to the correct location).


If you find this useful in your work, please cite the paper appropriately.
