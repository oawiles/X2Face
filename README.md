This repository contains the code for [X2Face](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html), presented at ECCV 2018.

The demo notebooks demonstrate the following:
- How to load the pre-trained models
- How to drive a face with another face in `./UnwrapMosaic/Face2Face_UnwrapMosaic.ipynb`
- How to edit the embedded face with a drawing or tattoo in `./UnwrapMosaic/Face2Face_UnwrapMosaic.ipynb`
- How to drive with pose in `./UnwrapMosaic/Pose2Face.ipynb`
- How to drive with audio in `./UnwrapMosiac/Audio2Face.ipynb`

To run the notebooks, you need:
- pytorch=0.2.0_4 
- torchvision
- PIL
- numpy
- matplotlib

It is **important** to use the right version of pytorch, as the defaults for sampling and some other things have changed in more recent versions of pytorch. In these cases, the pretrained models will not work properly.

Once the environment is set up, the pre-trained models can be downloaded from the [project page](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html) and the model paths in the notebooks updated appropriately (this should simply require setting the BASE_MODEL_PATH in the notebook to the correct location).


If you find this useful in your work, please cite the paper appropriately.
