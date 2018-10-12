This repository contains the code for [X2Face](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html), presented at ECCV 2018.

<h1>Demo Files for Trained Models</h1>

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

<h1>Training</h1>

Training code requires:
- tensorboardX

To train a model yourself, we have given an example training file using only the photometric loss.
To run this:
- Go to the [website](http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html)
- In the data section download the images and training/testing splits
- Update the paths in ./UnwrapMosaic/VoxCelebData_withmask.py
- Run the code with `python train_model.py --results_folder $WHERE_TO_SAVE_TENSORBOARD_FILES --model_epoch_path $WHERETOSAVEMODELS`

(Note that this can be run with any version of pytorch -- it is merely important that you train/test with the same version.)
