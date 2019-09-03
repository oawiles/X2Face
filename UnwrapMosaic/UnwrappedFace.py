import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from SkipNet import Pix2PixModel
from NoSkipNet_X2Face import Pix2PixModel as NoSkipPix2PixModel
from NoSkipNet_X2Face_pose import Pix2PixModel as NoSkipPix2PixModel_pose
import numpy as np

# Generates the architecture for the network
# E.g. the two images go from the given image to a sampler with values between -1,1 
# denoting the offset from the identity mapping.

class UnwrappedFaceWeightedAverage(nn.Module):
    def __init__(self, output_num_channels=2, input_num_channels=3, inner_nc=512):
        super(UnwrappedFaceWeightedAverage, self).__init__()
    
        self.pix2pixUnwrapped = Pix2PixModel(3)
    
        self.pix2pixSampler = NoSkipPix2PixModel(input_num_channels, output_num_channels, inner_nc=inner_nc)
    
    
    def forward(self, target_pose, *input_imgs):
        xs = np.linspace(-1,1,input_imgs[0].size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
    
        xs = torch.Tensor(xs).unsqueeze(0).repeat(input_imgs[0].size(0), 1,1,1).cuda()
    
        input_imgs_t = [0] * len(input_imgs)
        confidence = [0] * len(input_imgs)
        for i in range(0, len(input_imgs)):
                temp = self.pix2pixUnwrapped(input_imgs[i])[0]
                sampler = temp[:,0:2,:,:]
                confidence[i] = temp[:,2:3,:,:].unsqueeze(4).exp().add(0.1)
                input_imgs_t[i] = nn.Tanh()(sampler).permute(0,2,3,1) + Variable(xs, requires_grad=False)
                input_imgs_t[i] = nn.functional.grid_sample(input_imgs[i],  input_imgs_t[i]).unsqueeze(4)
                input_imgs_t[i] = input_imgs_t[i] * confidence[i].expand_as(input_imgs_t[i])
                
        # Combine multiple images
        input_imgs = torch.cat(input_imgs_t, 4)
        input_imgs = input_imgs.sum(4) 
    
        confidences = torch.cat(confidence, 4)
        confidences = confidences.sum(4)
    
        result_xc = input_imgs / confidences.expand_as(input_imgs)
        
        sampler = self.pix2pixSampler(target_pose)[0]
    
        
        if sampler.size(1) == 2:
                sampler_xy = nn.Tanh()(sampler)
                sampler_xy = sampler_xy.permute(0,2,3,1) + Variable(xs, requires_grad=False)
    
                sampled_image = nn.functional.grid_sample(result_xc, sampler_xy)
                return sampled_image
    
        sampler_xy = nn.Tanh()(sampler[:,0:2,:,:])
        #print(confidences.size(), xs.size())
        stddev = nn.Softplus().cuda()(sampler[:,2:,:,:]).clamp(max=40)
        sampler_xy = sampler_xy.permute(0,2,3,1) + stddev.permute(0,2,3,1).contiguous().mul(Variable(torch.randn(xs.size()).cuda(), requires_grad=False)) + Variable(xs, requires_grad=False) # choose values according to the std dev
        sampler_xy = sampler_xy.clamp(min=-1,max=1)
    
        sampled_image = nn.functional.grid_sample(result_xc, sampler_xy)
        return sampled_image, stddev
    
    def get_unwrapped_oneimage(self, input_img):
        xs = np.linspace(-1,1,input_img.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(input_img.size(0), 1,1,1).cuda()
    
        temp = self.pix2pixUnwrapped(input_img)[0]
        sampler = temp[:,0:2,:,:]
        sampler = nn.Tanh()(sampler).permute(0,2,3,1) + Variable(xs, requires_grad=False)
        input_img_t = nn.functional.grid_sample(input_img,  sampler)
        return input_img_t
    
    def get_unwrapped(self, *input_imgs):
        xs = np.linspace(-1,1,input_imgs[0].size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(input_imgs[0].size(0), 1,1,1).cuda()
    
        input_imgs_t = [0] * len(input_imgs)
        confidence = [0] * len(input_imgs)
        for i in range(0, len(input_imgs)):
                temp = self.pix2pixUnwrapped(input_imgs[i])[0]
                sampler = temp[:,0:2,:,:]
                confidence[i] = temp[:,2:3,:,:].unsqueeze(4).exp()
                input_imgs_t[i] = nn.Tanh()(sampler).permute(0,2,3,1) + Variable(xs, requires_grad=False)
                input_imgs_t[i] = nn.functional.grid_sample(input_imgs[i],  input_imgs_t[i]).unsqueeze(4)
                input_imgs_t[i] = input_imgs_t[i] * confidence[i].expand_as(input_imgs_t[i])
                
        # Combine multiple images
        input_imgs = torch.cat(input_imgs_t, 4)
        input_imgs = input_imgs.sum(4) 
    
        confidences = torch.cat(confidence, 4)
        confidences_sum = confidences.sum(4)
    
        result_xc = input_imgs / confidences_sum.expand_as(input_imgs)
        return result_xc, confidences
    
    def get_sampler(self, target_pose):
        sampler = nn.Tanh()(self.pix2pixSampler(target_pose)[0])
    
        xs = np.linspace(-1,1,sampler.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(target_pose.size(0), 1,1,1).cuda()
    
        sampler = sampler.permute(0,2,3,1) + Variable(xs, requires_grad=False)
        return sampler
    

class UnwrappedFaceWeightedAveragePose(nn.Module):
        def __init__(self, output_num_channels=2, input_num_channels=3, inner_nc=512, input_pose=True):
                super(UnwrappedFaceWeightedAveragePose, self).__init__()

                self.pix2pixUnwrapped = Pix2PixModel(3)
                self.pix2pixSampler = NoSkipPix2PixModel_pose(input_num_channels, output_num_channels, input_pose=input_pose, inner_nc=inner_nc)

        def forward(self, target_pose, pose_gt, *input_imgs):
                xs = np.linspace(-1,1,input_imgs[0].size(2))
                xs = np.meshgrid(xs, xs)
                xs = np.stack(xs, 2)

                xs = torch.Tensor(xs).unsqueeze(0).repeat(input_imgs[0].size(0), 1,1,1).cuda()

                input_imgs_t = [0] * len(input_imgs)
                confidence = [0] * len(input_imgs)
                for i in range(0, len(input_imgs)):
                        temp = self.pix2pixUnwrapped(input_imgs[i])[0]
                        sampler = temp[:,0:2,:,:]
                        confidence[i] = temp[:,2:3,:,:].unsqueeze(4).exp().add(0.1)
                        input_imgs_t[i] = nn.Tanh()(sampler).permute(0,2,3,1) + Variable(xs, requires_grad=False)
                        input_imgs_t[i] = nn.functional.grid_sample(input_imgs[i],  input_imgs_t[i]).unsqueeze(4)
                        input_imgs_t[i] = input_imgs_t[i] * confidence[i].expand_as(input_imgs_t[i])
                        
                # Combine multiple images
                input_imgs = torch.cat(input_imgs_t, 4)
                input_imgs = input_imgs.sum(4) 

                confidences = torch.cat(confidence, 4)
                confidences = confidences.sum(4)

                result_xc = input_imgs / confidences.expand_as(input_imgs)
                
                sampler = self.pix2pixSampler(target_pose, [pose_gt])[0]

                
                if sampler.size(1) == 2:
                        sampler_xy = nn.Tanh()(sampler)
                        sampler_xy = sampler_xy.permute(0,2,3,1) + Variable(xs, requires_grad=False)

                        sampled_image = nn.functional.grid_sample(result_xc, sampler_xy)
                        return sampled_image

                sampler_xy = nn.Tanh()(sampler[:,0:2,:,:])
                #print(confidences.size(), xs.size())
                stddev = nn.Softplus().cuda()(sampler[:,2:,:,:]).clamp(max=40)
                sampler_xy = sampler_xy.permute(0,2,3,1) + stddev.permute(0,2,3,1).contiguous().mul(Variable(torch.randn(xs.size()).cuda(), requires_grad=False)) + Variable(xs, requires_grad=False) # choose values according to the std dev
                sampler_xy = sampler_xy.clamp(min=-1,max=1)

                sampled_image = nn.functional.grid_sample(result_xc, sampler_xy)
                return sampled_image, stddev

        def get_unwrapped(self, *input_imgs):
                xs = np.linspace(-1,1,input_imgs[0].size(2))
                xs = np.meshgrid(xs, xs)
                xs = np.stack(xs, 2)
                xs = torch.Tensor(xs).unsqueeze(0).repeat(input_imgs[0].size(0), 1,1,1).cuda()

                input_imgs_t = [0] * len(input_imgs)
                confidence = [0] * len(input_imgs)
                for i in range(0, len(input_imgs)):
                        temp = self.pix2pixUnwrapped(input_imgs[i])[0]
                        sampler = temp[:,0:2,:,:]
                        confidence[i] = temp[:,2:3,:,:].unsqueeze(4).exp()
                        input_imgs_t[i] = nn.Tanh()(sampler).permute(0,2,3,1) + Variable(xs, requires_grad=False)
                        input_imgs_t[i] = nn.functional.grid_sample(input_imgs[i],  input_imgs_t[i]).unsqueeze(4)
                        input_imgs_t[i] = input_imgs_t[i] * confidence[i].expand_as(input_imgs_t[i])
                        
                # Combine multiple images
                input_imgs = torch.cat(input_imgs_t, 4)
                input_imgs = input_imgs.sum(4) 

                confidences = torch.cat(confidence, 4)
                confidences_sum = confidences.sum(4)

                result_xc = input_imgs / confidences_sum.expand_as(input_imgs)
                return result_xc, confidences

        def get_sampler(self, target_pose):
                sampler = nn.Tanh()(self.pix2pixSampler(target_pose)[0])

                xs = np.linspace(-1,1,sampler.size(2))
                xs = np.meshgrid(xs, xs)
                xs = np.stack(xs, 2)
                xs = torch.Tensor(xs).unsqueeze(0).repeat(target_pose.size(0), 1,1,1).cuda()

                sampler = sampler.permute(0,2,3,1) + Variable(xs, requires_grad=False)
                return sampler

class BottleneckFromNet(nn.Module):
    def __init__(self, output_num_channels=2, input_num_channels=3, inner_nc=128):
        super(BottleneckFromNet, self).__init__()
        self.pix2pixSampler = NoSkipPix2PixModel(input_num_channels, output_num_channels, inner_nc=inner_nc)

    def forward(self, target_pose, *input_img):
        bottleneck = self.pix2pixSampler(target_pose)[1]
        bottleneck = bottleneck.squeeze()
        out = bottleneck.detach()
        return out
