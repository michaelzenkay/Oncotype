import os
import numpy as np
import time
from os.path import join, dirname, basename
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil
from glob import glob
from ResnetModels import ResNet101
from DatasetGenerator import SegDatasetGenerator

#-------------------------------------------------------------------------------- 

class ClassNetTrainer():
    def __init__(self):
        self.lossMIN=100000
        self.resume = 0
    def addtolog(self, str):
        with open (self.log,'a') as f:
            f.write(str)

    ## ----------- Helper Fxns ------------------ ##
    def set_arch(self,arch,classes, init, pretrained=False, verbose=True, gpus=[0,1,2]):
        if init=='transfer':
            transfer=True
        else:
            transfer=None
        if arch =='res': model = ResNet101(classes, pretrained,transfer)
        model = torch.nn.DataParallel(model,device_ids=gpus)
        device = torch.device("cuda:0")
        model.to(device)
        if verbose==True:
            print('Using ' + arch + ' architecture')
        return model

    def set_optimizer(self,model,lr, opt='RMS'):
        return optim.Adam (model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # return optim.RMSprop(model.parameters(), lr=lr)

    def set_scheduler(self,optimizer, exp=False):
        # Exponential scheduler or Reduce on plateau
        if exp==True:
            return lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        else:
            return ReduceLROnPlateau(optimizer, factor=0.1, patience=4, mode='min', verbose=True)

    def set_loss(self):
        return nn.CrossEntropyLoss()

    def load_checkpoint(self, model, optimizer, nameCkpt,modelfn, verbose=True):
        self.resume = int(os.path.basename(modelfn).split(nameCkpt)[1].split('.')[0][1:])
        if verbose==True:
            print('resuming from ' + modelfn + ' epoch ' + str(self.resume))
        modelCheckpoint = torch.load(modelfn)
        model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
        optimizer.load_state_dict(modelCheckpoint['optimizer'])

    def warmstart(self, model, modelfn,verbose=True):
        if verbose==True:
            print('warmstarting from ' + modelfn + ' epoch ' + str(self.resume))
        modelCheckpoint = torch.load(modelfn)
        model.load_state_dict(modelCheckpoint['state_dict'], strict=False)

    def load_transfer(self,model,transfer,verbose=True):
        if not transfer==None:
            if verbose==True:
                print('Transfer learning from ' + transfer)
            modelTransfer = torch.load(transfer)
            model.load_state_dict(modelTransfer['state_dict'], strict=False)

    def write2csv(self,csv,text,mode='a'):
        with open(csv, mode) as fd:
            fd.write(text)

    ## ---------------------- Epoch ------------------ ##

    def epochTrain(self,model, dataLoader, optimizer, loss):
        # Trains one epoch on the model using the dataloader and calculating loss function with the given optimizer
        
        model.train()
        
        for batchID, (img, cla, imgfn) in enumerate (dataLoader):# , clafn, clin) in enumerate (dataLoader):
            # print(str(batchID) + '/' + str(dataLoader.__len__()))
            lbl_true = cla.cuda()
            lbl_probs = model.forward(img.cuda())
            lossvalue = loss(lbl_probs, lbl_true)
            # print(lossvalue)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
        
    def epochVal(self,model, dataLoader, optimizer, loss):
        # Calculates loss of the validation dataset
        
        model.eval()
        with torch.no_grad():
            outLoss = 0
            for i, (img, cla, imgfn) in enumerate(dataLoader):
                lbl_true = cla.cuda()
                lbl_probs = model.forward(img.cuda())
                lossvalue = loss(lbl_probs, lbl_true)
                outLoss += lossvalue.item()
        return outLoss


    ## -----------------Main Functions ---------##
    def train(self, trainfn, valfn, arch, pretrained, classes, batch, epochs, resize, crop, modelfn, init, pathCkpt,
              nameCkpt, lr, gpus=[0,1,2]):
        self.arch = arch
        self.log = join(pathCkpt, nameCkpt + '.log')

        # Model
        model = self.set_arch(arch, classes, init, pretrained, gpus=gpus)

        # Data-set/loader
        datasetTrain = SegDatasetGenerator(pathDatasetFile=trainfn)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch, shuffle=True, num_workers=6, pin_memory=True)
        datasetVal = SegDatasetGenerator(pathDatasetFile=valfn)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch, shuffle=True, num_workers=6, pin_memory=True)

        # Optimizers
        optimizer = self.set_optimizer(model, lr)
        scheduler = self.set_scheduler(optimizer)
        loss = self.set_loss()

        # Initialization
        if init == 'checkpoint':
            self.load_checkpoint(model, optimizer, nameCkpt, modelfn)
        elif init =='resume':
            self.resume=1
            modelfn = sorted(glob(join(pathCkpt,'*'+nameCkpt+'*')))[-1]
            self.load_checkpoint(model, optimizer, nameCkpt, modelfn)
        elif init == 'transfer':
            self.load_transfer(model, modelfn)
        elif init == 'warmstart':
            self.warmstart(model, modelfn)

        # Print Parameters
        print('Training size: ' + str(datasetTrain.__len__()))
        print('Validation size: ' + str(datasetVal.__len__()))

        for epochID in range(self.resume, epochs):
            # Train
            self.epochTrain(model, dataLoaderTrain, optimizer, loss)

            # Val
            lossVal = self.epochVal(model, dataLoaderVal, optimizer, loss)
            self.addtolog('Val Loss for Epoch ' + str(epochID) + ' is ' + str(lossVal) + '\n')

            # Step
            scheduler.step(lossVal)

            # If resuming, find loss of the last epoch that we are resuming from
            if epochID == self.resume and epochID > 1:
                self.lossMIN = lossVal
                continue

            # If better val loss, save
            if lossVal < self.lossMIN:
                self.lossMIN = lossVal
                savename = os.path.join(pathCkpt, nameCkpt + '_' + str(epochID + 1).zfill(2) + '.pth')
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': self.lossMIN,
                            'optimizer': optimizer.state_dict()}, savename)  # 'm-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + ']  ' + str(lossVal))

    def inference(self,inference_csv, arch, classes, batch, resize, crop, modelfn, csv_out, gpu):
        # csv_out is results file, will contain the follwing results:
        
        if gpu==True:
            cudnn.benchmark = True
            device = torch.device("cuda:0")
        else:
            device= torch.device("cpu")
        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD

        if arch == 'res':
            model = ResNet101(classes, False)

        if gpu==True:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
            modelCheckpoint = torch.load(modelfn)
            model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            modelCheckpoint = torch.load(modelfn)
            model.load_state_dict(modelCheckpoint['state_dict'])
        model.to(device)

        datasetInf = SegDatasetGenerator(pathDatasetFile=inference_csv,inference=True)
        dataLoaderInf = DataLoader(dataset=datasetInf, batch_size=batch, num_workers=1, shuffle=False, pin_memory=True)

        model.eval()
        with open(csv_out, 'a') as fd:
            fd.write('imgfn,prob0,prob1,lbl_pred\n')
            for i, (img, imgfn) in enumerate(dataLoaderInf):
                lbl_prob = model(img.to(device))
                lbl_prob_cpu = lbl_prob.cpu().detach()
                for n in range(0, len(imgfn)):
                    prob0 = float(lbl_prob_cpu[n][0])
                    prob1 = float(lbl_prob_cpu[n][1])
                    lbl_pred = int(np.argmax(lbl_prob_cpu[n]))
                    fd.write(imgfn[n] + ',' + str(prob0) + ',' + str(prob1) + ',' + str(lbl_pred) + '\n')