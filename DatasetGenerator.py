import os
import numpy as np
from PIL import Image
import PIL
import pandas
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import cv2

def bbox(seg, dims=1):
    # Find extents of segmentation as a box with dimensions that are divisible by 'dim'
    x_indices = np.where(np.any(seg[:, :], axis=0))[0]
    y_indices = np.where(np.any(seg[:, :], axis=1))[0]

    if x_indices.shape[0]:

        x1, x2 = x_indices[[0, -1]]
        y1, y2 = y_indices[[0, -1]]

        # Increment x2 and y2 by 1 since they're not part of the initial box
        x2, y2 = [x2 + 1, y2 + 1]

        # Minimum crop size (multiple of dims)
        dim = dims * max(math.ceil((x2 - x1) / dims), math.ceil((y2 - y1) / dims))

        # Find bbox with dim x dim crop size
        x1, x2 = [int((x1 + x2) / 2 - dim / 2), int((x1 + x2) / 2 + dim / 2)]
        y1, y2 = [int((y1 + y2) / 2 - dim / 2), int((y1 + y2) / 2 + dim / 2)]

        # Boundary Conditions
        if (x1 + x2) / 2 - dim / 2 < 0:
            x1, x2 = [0, dim]
        if (x1 + x2) / 2 + dim / 2 > seg.shape[0]:
            x1, x2 = [seg.shape[0] - dim, seg.shape[0]]
        if (y1 + y2) / 2 - dim / 2 < 0:
            y1, y2 = [0, dim]
        if (y1 + y2) / 2 + dim / 2 > seg.shape[1]:
            y1, y2 = [seg.shape[1] - dim, seg.shape[1]]

    else:
        x1, x2, y1, y2 = 0, 0, 0, 0

    return x1, x2, y1, y2

def create_MRI_breast_mask(image, threshold=15, size_denominator=55):
    """
    Creates a rough mask of breast tissue returned as 1 = breast 0 = nothing
    :param image: the input volume (3D numpy array)
    :param threshold: what value to threshold the mask
    :param size_denominator: The bigger this is the smaller the structuring element
    :return: mask: the mask volume
    """
    tmp = image.convert('L')
    # Create the mask
    mask2 = np.copy(np.squeeze(tmp))

    # Apply gaussian blur to smooth the image
    mask2 = cv2.bilateralFilter(mask2.astype(np.float32),20,75,75)

    # Threshold the image. Change to the bool
    mask1 = np.squeeze(mask2 < threshold).astype(np.int16)

    # Define the CV2 structuring element
    radius_close = np.round(mask1.shape[1] / size_denominator).astype('int16')
    kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

    # Apply morph close
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel_close)

    # Invert mask
    mask1 = ~mask1

    # Add 2
    mask1 += 2

    masked = np.copy(np.squeeze(image))
    for a in range(0,np.array(image).shape[2]):
        masked[:,:,a] = masked[:,:,a]*mask1

    return Image.fromarray(masked)

class DatasetGenerator(Dataset):

    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathDatasetFile, augment=True, inference=False):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.listLabelPaths = []
        self.inference = inference
        self.augment = augment
        #---- Open file, get image paths and labels

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

            if self.inference==False:
                self.listLabelPaths.append(str(entry[0]))
                self.listImageLabels.append(int(entry[1])) # Oncotype class

    #-------------------------------------------------------------------------------- 


    #--------------------------------------------------------------------------------
    def transform(self, image):
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)

        if self.augment==True:
            # Random rotate
            if random.random() > 0.5:
                min_angle = -15
                max_angle = 15
                angle = random.randrange(min_angle,max_angle)
                image = TF.rotate(image, angle)

            # Random zoom crop
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(224/2, 224/2))
                image = TF.crop(image, i, j, h, w)
                resize = transforms.Resize(size=(224, 224))
                image = resize(image)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)


            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)

            # Random hue
            if random.random() > 0.5:
                image = TF.adjust_hue(image, random.randrange(-10, 10)/100)

            # Change Saturation
            if random.random() > 0.5:
                min_sat, max_sat = 90,110
                image = TF.adjust_saturation(image, random.randrange(min_sat, max_sat)/100)

        # Transform to tensor
        image = TF.to_tensor(image)

        # Normalize
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        return image

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageData = create_MRI_breast_mask(imageData)

        # Training and Validation
        if self.inference == False:

            # Oncotype
            imageLabel= self.listImageLabels[index]

            # Transform
            imageData = self.transform(imageData)

            return imageData, imageLabel, imagePath

        # Inference
        else:
            tform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
            imageData = tform(imageData)
            imageData = TF.normalize(imageData, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

            return imageData, imagePath
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #--------------------------------------------------------------------------------

class SegDatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathDatasetFile, augment=True, inference=False):

        # Initialize
        self.listImagePaths = []
        self.listSegPaths = []
        self.listImageLabels = []
        self.inference = inference
        self.augment = augment

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

            if self.inference == False:
                self.listSegPaths.append(str(entry[1]))
                self.listImageLabels.append(int(entry[2]))  # Oncotype

    #--------------------------------------------------------------------------------
    def transform(self, image, seg, scale = 3):
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)


        if self.augment==True:
            # Center Crop
            if random.random() > 0:#.5:
                # Random scaled crop

                # scale = max scale, so find a random scale between 1 to max
                if scale!=1:
                    scale = random.randrange(1,scale)

                # Find random patch
                if scale >= 2:
                    x1, x2, y1, y2 = imgtools.bbox(seg, dims=128)
                    w = y2-y1
                    h = x2-x1
                    j = x1
                    i = y1
                    image = TF.crop(image, i, j, h, w)
                    image = resize(image)
                    scale = scale/2

                # Random affine
                min_angle = -15
                max_angle = 15
                angle = random.randrange(min_angle,max_angle)
                translate = (0,0)
                min_shear = -5
                max_shear = 5
                shear = random.randrange(min_shear, max_shear)
                image = TF.affine(image, angle, translate, scale, shear, resample=PIL.Image.BICUBIC)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)

            # Random hue
            if random.random() > 0.5:
                image = TF.adjust_hue(image, random.randrange(-10, 10)/100)

            # Change Saturation
            if random.random() > 0.5:
                min_sat, max_sat = 90,110
                image = TF.adjust_saturation(image, random.randrange(min_sat, max_sat)/100)

        # Transform to tensor
        image = TF.to_tensor(image)

        # Normalize
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        return image

    def __getitem__(self, index):
        # Image
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageData = create_MRI_breast_mask(imageData)

        # Seg
        segPath = self.listSegPaths[index]
        segData = np.load(segPath)

        # Training and Validation
        if self.inference == False:
            # Oncotype
            imageLabel = self.listImageLabels[index]

            # Transform
            imageData = self.transform(imageData,segData)

            return imageData, imageLabel, imagePath

        # Inference
        else:
            imageData, null = self.transform(imageData, imageData.convert('L'))
            return imageData, imagePath

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listImagePaths)

class camDatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathDatasetFile, transform):

        self.listImagePaths = []
        self.transform = transform

        # ---- Open file, get image paths and labels

        csv = pandas.read_csv(pathDatasetFile)
        for entry in csv.values:
            imagePath = str(entry[0])

            self.listImagePaths.append(imagePath)

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')

        if self.transform != None: imageData = self.transform(imageData)

        return imageData,  imagePath

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listImagePaths)

# --------------------------------------------------------------------------------
