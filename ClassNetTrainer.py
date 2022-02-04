import numpy as np
import torch
from torch.utils.data import DataLoader
from resnet3d import resnet18, resnet50, resnet101
from DatasetGenerator import SegDatasetGenerator as DatasetGenerator

class ClassNetTrainer():
    def __init__(self):
        self.resume=0
    ## ----------- Helper Fxns ------------------ ##
    def set_arch(self, arch, classes, pretrained=False, verbose=True, gpus=[0,1,2]):
        if arch =='res': 
            model = resnet50(classes, pretrained)
        else: 
            print('Not supported architecture')

        # Set model to parallel on the gpus, primary gpu is gpus[0]
        if not gpus == None:
            model = torch.nn.DataParallel(model,device_ids=gpus)
            torch.cuda.set_device(gpus[0])
            self.device = torch.device("cuda")
        
        # Use CPU if no cuda devices
        else:
            self.device = torch.device("cpu")
        model.to(self.device)
        
        if verbose==True:
            print('Using ' + arch + ' architecture')
        return model

    def load_model(self, model, modelfn,verbose=True,strict=True):
        if verbose==True:
            print('loading model from ' + modelfn)
        model.load_state_dict(torch.load(modelfn,map_location=self.device)['state_dict'],strict=strict)
        return model
    
    def write2csv(self,csv,text,mode='a'):
        with open(csv, mode) as fd:
            fd.write(text)
            
    def inference(self,inference_csv, arch, classes, batch, resize, crop, modelfn, csv_out, gpu):
        # csv_out is results file, will contain the follwing results:
        # imgfn, prob0, prob1, lbl_pred
        # for each pngfn in inference_csv
        
        model = self.set_arch(arch,classes,gpus=gpu)
        
        # Load Model
        model = self.load_model(model, modelfn,verbose=True)

        datasetInf = DatasetGenerator(pathDatasetFile=inference_csv,inference=True)
        dataLoaderInf = DataLoader(dataset=datasetInf, batch_size=batch, num_workers=1, shuffle=False, pin_memory=True)

        model.eval()
        print('saving inferences to ' + csv_out)
        self.write2csv(csv_out,'imgfn,prob0,prob1,lbl_pred\n', mode='w')
        for i, (img, imgfn, extra) in enumerate(dataLoaderInf):
            lbl_prob = model.forward(img.cuda(), extra)
            lbl_prob_cpu = lbl_prob.cpu().detach()
            for n in range(0, len(imgfn)):
                prob0 = float(lbl_prob_cpu[n][0])
                prob1 = float(lbl_prob_cpu[n][1])
                lbl_pred = int(np.argmax(lbl_prob_cpu[n]))
                self.write2csv(csv_out,
                    imgfn[n] + ',' + str(prob0)+ ',' + str(prob1) + ',' + str(lbl_pred) + '\n')

    torch.cuda.empty_cache()