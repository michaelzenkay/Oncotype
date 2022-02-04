import nibabel as nib
import os
import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import scipy.ndimage
    
 #-------------------------------------------------------------------------------- 

class SegDatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathDatasetFile, inference=False):

        # Initialize
        self.listImagePaths = []
        self.listSegPaths = []
        self.listImageLabels = []
        self.inference = inference

        # ---- Grab image and seg paths

        if isinstance(pathDatasetFile, list):
            frame = []
            for i in range(0, len(pathDatasetFile)):
                df = pandas.read_csv(pathDatasetFile[i])
                frame.append(df)
            csv = pandas.concat(frame, axis=0, ignore_index=True)

        else:
            csv = pandas.read_csv(pathDatasetFile)

        for entry in csv.values:
            self.listImagePaths.append(str(entry[0]))

    #--------------------------------------------------------------------------------
    def transform(self, image, seg):
        
        # Transform to tensor
        image =  (image - np.average(image))/np.std(image)
        image = TF.to_tensor(image)

        # image = TF.Normalize(image,mean=[0.5, 0.5, 0.5],std=[0.25, 0.25, 0.25])

        return image

    def __getitem__(self, index):
        # Image
        imagePath = self.listImagePaths[index]
        sz = nib.load(imagePath).shape
        if self.inference==True:
            # Load
            prefix = imagePath.split('.nii.gz')[0][:-1]
            img = np.zeros((sz[0],sz[1],sz[2],3))
            img[:,:,:,0] = nib.load(prefix + '1.nii.gz').get_fdata()
            img[:,:,:,1] = nib.load(prefix + '2.nii.gz').get_fdata()
            img[:,:,:,2] = nib.load(prefix + '3.nii.gz').get_fdata()
            seg = nib.load(prefix + 'L.nii.gz').get_fdata()
            
            dim=[64,64,64]
            # Get Tumor Center
            d=np.zeros((3,2))
            center = scipy.ndimage.measurements.center_of_mass(seg)
            for i in range(0,d.shape[0]):
                # Find the edges by going from dim/2 from center 
                d[i,0] = int(center[i] - dim[i]/2)            # Low side
                d[i,1] = int(d[i,0] + dim[i])                 # High side
                # Evaluate Boundary Conditions
                if d[i,0]<0:                                  # If low side is negative
                    d[i, 0] = int(0)                            # Set low side to 0
                    d[i, 1] = dim[i]         
                if d[i,1]>seg.shape[i]:            # If high side is more than input
                    d[i, 1] = seg.shape[i]         # Set high side to high extent of segmentation
                    d[i, 0] = seg.shape[i] - dim[i]# Set low sie to high extent - dim[i]
            # 3d Crop
            d = d.astype('int')
            segData = seg[d[0,0]:d[0,1],d[1,0]:d[1,1],d[2,0]:d[2,1]]
            img = img[d[0,0]:d[0,1],d[1,0]:d[1,1],d[2,0]:d[2,1]]
            imageData = img/img.max()
            
            
        else:
            segPath = self.listSegPaths[index]
            imageData = nib.load(imagePath).get_data()
            imageData = imageData/imageData.max()
            segData = nib.load(segPath).get_data()

        # Tumor SER and size
        masked = imageData[segData > 0]
        s0 = np.average(masked[:, 0])
        s1 = np.average(masked[:, 1])
        s2 = np.average(masked[:, 2])
        ser = abs((s1-s0)/(s2-s0 + 0.00001))
        size = int(sum(sum(sum(segData))))
        if np.isnan(size):
            size=0
        extra = np.zeros(2)
        extra[0] = float(ser)
        extra[1] = float(size/10000)

        # Transform
        imageData = torch.tensor(imageData).permute(3,0,1,2).type(torch.FloatTensor)
            
        # Training and Validation
        if self.inference == False:
            # Oncotype
            imageLabel = self.listImageLabels[index]
            return imageData, imageLabel, imagePath, extra

        # Inference
        else:
            return imageData, imagePath, extra

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listImagePaths)
