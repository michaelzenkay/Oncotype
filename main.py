from glob import glob
import os
from os.path import join, dirname, basename
import preprocess
from ClassNetTrainer import ClassNetTrainer

#--------------------------------------------------------------------------------
def main (infer=False):
    arch = 'res'
    modelfn = '../models/odx.pth'
    nameCkpt = 'save_name'
    pathNii = '../imgs'
    pathPreprocessed = '../preprocessed'
    pathDataset = './dataset/'
    pathCkpt = join(pathDataset,nameCkpt)
    csvFns = './imgfns.csv'
    
    # Preprocessing
    preprocess.preprocess_dir(pathNii, pathPreprocessed, csvFns)
    
    # K fold Splitting
    if infer==False:
        splitter.k_fold(csvFns,pathDataset)
    
    ## Initialization from a model
    # init = [None, 'checkpoint', 'transfer', 'warmstart']
    init = 'warmstart'

    # Hyperparameters
    classes = 2
    batch = 50
    resize = 256
    crop = 224
    folds = 5
    lr = 0.0005

## TRAIN
    runTrain(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt, modelfn=modelfn, init=init, lr = lr)

## INFER
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
