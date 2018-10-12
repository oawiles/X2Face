# DISCLAIMER: this is a easy to use + slimmed down + refactored version of the training code used in the ECCV paper: X2Face
# It should give approximately similar results to what is in the paper (e.g. the frontalised unwrapped face
# and that the driving portion of the network transforms this frontalised face into the given view).
# It should also give a good idea of how to train the network.

# (c) Olivia Wiles

from VoxCelebData_withmask import VoxCeleb

import shutil
import os
import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.transforms import ToTensor, Scale, Compose
import torch.optim as optim
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage

parser = argparse.ArgumentParser(description='UnwrappedFace')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--sampler_lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--threads', type=int, default=1, help='Num Threads')
parser.add_argument('--batchSize', type=int, default=16, help='Batch Size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--num_views', type=int, default=2, help='Num views')
parser.add_argument('--copy_weights', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='UnwrappedFaceSampler_from1view')
parser.add_argument('--inner_nc', type=int, default=128)
parser.add_argument('--old_model', type=str, default='')
parser.add_argument('--results_folder', type=str, default='/scratch/local/hdd/ow/results/') # Where temp results will be stored
parser.add_argument('--model_epoch_path', type=str, default='/scratch/local/hdd/ow/faces/models/python/sampler/%s/', help='Location to save to')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

writer = SummaryWriter(opt.results_folder)

opt.model_epoch_path = opt.model_epoch_path % 'x2face'

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,inner_nc=opt.inner_nc)

if opt.copy_weights:
        checkpoint_file = torch.load(opt.old_model)
        model.load_state_dict(checkpoint_file['state_dict'])
        opt.model_epoch_path = opt.model_epoch_path + 'copyWeights'
        del checkpoint_file


criterion = nn.L1Loss()

model = model.cuda()

criterion = criterion.cuda()
parameters = [{'params' : model.parameters()}]
optimizer = optim.SGD(parameters, lr=opt.lr, momentum=0.9)

def run_batch(imgs):
        return model(imgs[1].cuda(), (imgs[0].cuda())), imgs

def get_unwrapped(imgs):
        return model.get_unwrapped_oneimage(imgs[0].cuda())

def train(epoch, num_views):
        train_set = VoxCeleb(num_views, epoch, 1)

        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        epoch_train_loss = 0

        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
                result, inputs = run_batch(batch[0])

                loss = criterion(result, inputs[opt.num_views-1].cuda())

                optimizer.zero_grad()
                epoch_train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                loss_mean = epoch_train_loss / iteration
                if iteration % 1000 == 0 or iteration == 1:
                        for i in range(0, len(inputs)):
                                input = inputs[i]
                                if input.size(1) == 2:
                                        writer.add_image('Train/img_dim%d_%d1' % (i, iteration), input[:,0:1,:,:].data.cpu(), epoch)
                                        writer.add_image('Train/img_dim%d_%d2' % (i, iteration), input[:,1:2,:,:].data.cpu(), epoch)
                                else:
                                        writer.add_image('Train/img%d_%d1' % (i, iteration), input.data.cpu(), epoch)

                        writer.add_image('Train/result%d' % (iteration), result.data.cpu(), epoch)
                        writer.add_image('Train/gt%d' % (iteration), inputs[opt.num_views-1].data.cpu(), epoch)

                        unwrapped = get_unwrapped(batch[0])
                        writer.add_image('Train/unwrapped%d' % (iteration), unwrapped.data.cpu(), epoch)


                print("===> Train Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                      len(training_data_loader), loss_mean))

                if iteration == 2000: # So we can see faster what's happening
                        break
        return epoch_train_loss / iteration

def val(epoch, num_views):
        val_set = VoxCeleb(num_views, 0, 2)

        validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

        model.eval()
        epoch_val_loss = 0

        for iteration, batch in enumerate(validation_data_loader, 1):
                result, inputs = run_batch(batch[0])
                loss = criterion(result, inputs[opt.num_views-1].cuda())

                epoch_val_loss += loss.data[0]
                loss_mean = epoch_val_loss / iteration


                if iteration % 1000 == 0 or iteration == 1:
                        for i in range(0, len(inputs)):
                                input = inputs[i]
                                if input.size(1) == 2:
                                        writer.add_image('Val/img_dim%d_%d1' % (i, iteration), input[:,0:1,:,:].data.cpu(), epoch)
                                        writer.add_image('Val/img_dim%d_%d2' % (i, iteration), input[:,1:2,:,:].data.cpu(), epoch)
                                else:
                                        writer.add_image('Val/img%d_%d1' % (i, iteration), input.data.cpu(), epoch)

                        writer.add_image('Val/result%d' % (iteration), result.data.cpu(), epoch)
                        writer.add_image('Val/gt%d' % (iteration), inputs[opt.num_views-1].data.cpu(), epoch)
                        unwrapped = get_unwrapped(batch[0])
                        writer.add_image('Val/unwrapped%d' % (iteration), unwrapped.data.cpu(), epoch)

                print("===> Val Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                              len(validation_data_loader), loss_mean))

                if iteration == 2000: # So we can see faster what's happening
                        break

        return epoch_val_loss / iteration

def checkpoint(model, epoch):
        dict = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}

        model_out_path = "{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch)

        if not(os.path.exists(opt.model_epoch_path)):
                os.makedirs(opt.model_epoch_path)
        torch.save(dict, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

        for i in range(0, epoch-1):
                if os.path.exists("{}model_epoch_{}.pth".format(opt.model_epoch_path, i)):
                        os.remove( "{}model_epoch_{}.pth".format(opt.model_epoch_path, i))



if opt.copy_weights:
        checkpoint_file = torch.load(opt.old_model)
        model.load_state_dict(checkpoint_file['state_dict'])

start_epoch = opt.start_epoch
for epoch in range(start_epoch, 3000):
        if epoch > 0:
                checkpoint_file = torch.load("{}model_epoch_{}.pth".format(opt.model_epoch_path, epoch-1))
                model.load_state_dict(checkpoint_file['state_dict'])
                optimizer.load_state_dict(checkpoint_file['optimizer'])

        tloss = train(epoch, opt.num_views)
        with torch.no_grad():
                vloss = val(epoch, opt.num_views)

        writer.add_scalars('TrainVal/loss', {'train' : tloss, 'val' : vloss}, epoch)
        checkpoint(model, epoch)

