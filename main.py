from glob import glob
import os
from os.path import join, dirname, basename

from ClassNetTrainer import ClassNetTrainer

#--------------------------------------------------------------------------------
def main ():
    arch = 'res'
    modelfn = './dense121_wk20201031.pth'
    nameCkpt = 'save_name'

    ## Initialization from a model
    # init = [None, 'checkpoint', 'transfer', 'warmstart']
    init = 'warmstart'

    # Dataset
    pathDataset = 'path/to/dataset'

    # Save
    pathCkpt = pathDataset
    classes = 2
    batch = 50
    resize = 256
    crop = 224
    folds = 5
    lr = 0.0005

## TRAIN - ispy_T1 + weak
    runTrain(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt, modelfn=modelfn, init=init, lr = lr)

## INFER - ispy_T2,3,4
    csv = 'path/to/inference.csv'
    csv_out = 'path/to/output.csv'

    network = ClassNetTrainer()
    network.inference(csv, arch, classes, batch, resize, crop, modelfn, csv_out,gpu=True)

#--------------------------------------------------------------------------------
def runTrain(arch, classes, batch, resize, crop, folds, pathDataset,pathCkpt, nameCkpt, modelfn=None,init=None, lr = 0.00001, epochs = 100, pretrain=True):
    for f in range(1, folds+1):
        # Dataset
        trainfn = join(pathDataset, 'train_' + str(f) +'.csv')
        valfn = join(pathDataset, 'val_' + str(f) + '.csv')

        # Save
        NameCkpt = nameCkpt + str(f)

        network = ClassNetTrainer()
        network.train(trainfn, valfn, arch, pretrain, classes, batch, epochs, resize, crop, modelfn, init, pathCkpt, NameCkpt, lr)

#--------------------------------------------------------------------------------
def Infer(csv, modelfn, batch=50, csv_out=None, gpu=True):
    classes = 2
    resize = 256
    crop = 256
    arch = 'res'
    if csv_out==None:
        csv_out = join(dirname(csv), 'inferred.csv')
    network = ClassNetTrainer()
    print('Inferring from ' + csv + ' \nUsing model ' + modelfn + ' \nSaving to ' + csv_out)
    network.inference(csv, arch, classes, batch, resize, crop, modelfn, csv_out, gpu=gpu)

#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
