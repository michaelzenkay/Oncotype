from glob import glob
import os
from os.path import join, dirname, basename
from ClassNetTrainer import ClassNetTrainer
from axplot import inference_plotter
from roc import roc

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

## TEST - ispy_T1
    runTest(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt)
    ROC_vis(pathDataset, pathCkpt, nameCkpt, folds)

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
def runTest(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt):
    testfn = join(pathDataset, 'test.csv')
    for fold in range(1, folds+1):
        modelfn = sorted(glob(join(pathCkpt, nameCkpt + str(fold) + '*.pth')))[-1]
        csv_out = join(dirname(modelfn), 'results_' + str(fold) + '_' + nameCkpt + '.csv')
        network = ClassNetTrainer()
        print('Testing with ' + testfn + ' on ' + modelfn + ' saving to ' + csv_out)
        network.test(testfn, modelfn, arch, classes, batch, resize, crop, csv_out)

#--------------------------------------------------------------------------------
def ROC_vis(pathDataset,pathCkpt,nameCkpt,folds):
    rocfile = join(pathDataset,'roc.png')
    analysisfile = join(pathDataset,'analysis_wln.csv')
    print('Running ROC Analysis at ' + analysisfile + ', image to ' + rocfile)
    results_csvs=glob(join(pathCkpt, 'results_*_' + nameCkpt + '*.csv'))
    roc(results_csvs,rocfile,analysisfile,folds)

    for fold in range(1,folds+1):
        output = join(pathDataset, 'result_fold' + str(fold).zfill(1))
        if not os.path.exists(output):
            os.mkdir(output, mode=0o777)
        print('plotting results to ' + output)
        plotter = inference_plotter(results_csvs[fold-1],output)
        plotter.plot_csv_results()

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
