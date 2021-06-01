import os
from glob import glob
import numpy as np
import nibabel as nib
from os.path import join, basename, dirname,exists
import csv
import cv2
import matplotlib.pyplot as plt

class class_pat():
    def __init__(self, fn, test=False):
        self.fn = fn
        self.id = self.get_id(fn)
        self.path = os.path.dirname(fn)
        self.segpath = join(os.path.dirname(os.path.dirname(fn)),'seg')
        self.test = test
    def get_id(self,fn):
        return basename(fn).split('_')[0] + '_' + basename(fn).split('_')[1]
    def get_img(self):
        self.img = nib.load(self.fn).get_data()
        self.dyns = 1
    def get_3dyn(self):
        sum = join(self.path, self.id)
        # Find 1nd and 3rd dynamic
        try:
            img1fn = glob(join(sum + '*1.nii*'))[0]
            img2fn = glob(join(sum + '*2.nii*'))[0]
            img3fn = glob(join(sum + '*3.nii*'))[0]

            img1 = nib.load(img1fn).get_data()
            img2 = nib.load(img2fn).get_data()
            img3 = nib.load(img3fn).get_data()

            # img = 4d numpy : x,y,z,d where d is dynamic
            self.img = np.stack((img1, img2, img3), axis=3)
            self.dyns = 3
        except:
            print(self.fn + ' does not have 3 dynamics')
            self.dyns = 1
    def get_seg(self):
        try:
            # Get bpe label if it exists
            segs = glob(join(self.path, self.id + '*B*'))
            if len(segs)==0:
                segs = glob(join(self.path, self.id + '*L*'))
                if len(segs)==0:
                    segs = glob(join(self.path, self.id + '*mask*'))
            self.seg = nib.load(segs[0]).get_data()
            self.segs = 1
        except:
            print('no seg for ' + self.fn)
            self.segs = 0
    def normalizebeforepngexport(self, percentile = 99.9):
        """
        TMP is just the tumor region masked by self.seg
        this function normalizes self.img based on the percentile of TMP (winzorization)
        :return:
        """
        try:
            # If we have segmentations, normalize based on winsorized roi
            tmp = self.img*np.repeat(self.seg[:,:,:,np.newaxis],3,axis=3)
            perc = np.percentile(tmp[tmp.nonzero()], percentile)
        except:
            # If we dont have segmentations, normalize based on whole image
            perc = np.percentile(self.img,percentile)
        self.img = (self.img-self.img.min())/perc
        self.img[self.img>1] = 1
    def exportpng0csv(self, out, csv,overwrite=False):
        self.normalizebeforepngexport()
        for x in range(self.img.shape[0]):
            # Save slice image
            imgfn = self.id + '_' + str(x).zfill(3) + '.png'
            imgfn = join(out, imgfn)
            if not exists(imgfn) or overwrite==True:
                plt.imsave(imgfn, self.img[x, :, :, :])
                # Save img, slice fns and label to csv
                if not csv==None:
                    with open(csv, 'a') as fd:
                        fd.write(imgfn + ',\n')
    def exportpngseg0csv(self,out,csv):
        self.normalizebeforepngexport()
        for x in range(self.img.shape[0]):
            # Save slice image
            imgfn = self.id + '_' + str(x).zfill(3) + '.png'
            imgfn = join(out, 'img', imgfn)
            if not exists(dirname(imgfn)):
                os.makedirs(dirname(imgfn),exist_ok=True)
            plt.imsave(imgfn, self.img[x, :, :, :])

            # Save slice segmentation
            lblfn = self.id + '_' + str(x).zfill(3) + '.npy'
            lblfn = join(out, 'cla', lblfn)
            if not exists(dirname(lblfn)):
                os.makedirs(dirname(lblfn),exist_ok=True)
            np.save(lblfn, self.seg[x, :, :])
            
            # Determine slice label
            if np.sum(self.seg[x,:,:]) > 0:
                label = 1
            else:
                label = 0

            # Save img, slice fns and label to csv
            if not exists(dirname(csv)):
                os.makedirs(dirname(csv), exist_ok=True)
            with open(csv, 'a') as fd:
                fd.write(imgfn + ',' + lblfn + ',' + str(label) +',\n')
    def reshape(self,shape=256):
        print('start')
        print(self.img.shape)
        # If not the desired shape, find the scale factor
        if self.img.shape[2] != shape:
            scale = shape/self.img.shape[2]
            if scale>0 and self.img.shape[1] != self.img.shape[2]:
            # zero pad
                w1 = int((shape - self.img.shape[2])/2)
                w2 = int((shape + self.img.shape[2])/2)
                multichannelimg = np.zeros((self.img.shape[0],self.img.shape[1],shape, self.img.shape[3]))
                seg = np.zeros((self.img.shape[0],self.img.shape[1],shape))
                try:
                    multichannelimg[:,:,w1:w2,:] = self.img
                    seg[:,:,w1:w2] = self.seg
                except:
                    print('error')
            else:
                #scale
                tmp  = cv2.resize(self.img[1,:,:,:], None, fx=scale, fy=scale)
                multichannelimg = np.zeros((self.img.shape[0],tmp.shape[0],tmp.shape[1], tmp.shape[2]))
                try:
                    seg = np.zeros((self.seg.shape[0],tmp.shape[0],tmp.shape[1]))
                    self.seg = seg
                except:
                    pass
                for i in range(self.img.shape[0]):
                    multichannelimg[i,:,:,:] = cv2.resize(self.img[i,:,:,:], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    try:
                        seg[i,:,:] = cv2.resize(self.seg[i,:,:], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    except:
                        pass
            self.img = multichannelimg
            try:
                self.seg = seg
            except:
                pass
            print('scaled to ')
            print(self.img.shape)

        h = self.img.shape[1]
        if h != shape:
            h1 = int((h - shape) / 2)
            h2 = int((h + shape) / 2)
            tmpimg= self.img[:, h1:h2, :,:]
            self.img = tmpimg
            try:
                tmpseg = self.seg[:, h1:h2, :]
                self.seg = tmpseg
            except:
                pass
            print('cropped to ')
            print(self.img.shape)
def preprocess_dir(indir,outdir,out_csv=None,shape=256,overwrite=True):
    pats = glob(join(indir, '*2.nii*'))
    for pat in pats:
        preprocess_single(pat,outdir,out_csv=out_csv,shape=shape,overwrite=overwrite)
def preprocess_single(imgfn,outdir,out_csv=None, shape=256, overwrite=False):
    if out_csv==None:
        out_csv = join(outdir,'fns.csv')
    patobj = class_pat(imgfn)
    patobj.get_3dyn()
    patobj.get_seg()
    if patobj.dyns >= 3:
        if hasattr(patobj,'img'):
            patobj.reshape(shape=shape)
            # If segmentations are available, export pngs and segmentations 
            if patobj.segs == 1:
                patobj.exportpngseg0csv(outdir,out_csv)
            # If no segmentations are available, export pngs 
            else:
                patobj.exportpng0csv(outdir,out_csv,overwrite)
        else:
            print('no img patobj')
    else:
        print('Less than 3 dynamics available for ' + imgfn)
        pass
        # patobj.__delete__()