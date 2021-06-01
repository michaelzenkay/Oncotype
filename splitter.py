import csv
import pandas
import numpy as np
from glob import glob
import os
from os.path import basename, dirname, join, exists
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit

def get_id (fn):
    return basename(fn).split('_')[0] + basename(fn).split('_')[1]

def pattocsv(csv_out,csv_data,tvt_pats,pats):
    # Outputs a csv with full filename of img,lbl,cla
    if not exists(dirname(csv_out)):
        os.makedirs(dirname(csv_out, exist_ok=True))
    with open(csv_out, 'a') as fd:
        # For each patient that is going in the training set
        for pat in tvt_pats:
            # What indices in original csv?
            indices = np.where(pats == pat)[0]
            for index in indices:
                # Grab the whole row
                data = csv_data.values[index]
                imgfn = str(data[0])
                segfn = str(data[1])
                label = str(data[2])
                # Dump to csv file
                fd.write(imgfn + ',' + segfn + ',' + label + '\n')

def split_data(csv, out,train_val_test=(0.7,0.2,0.1), shuffle=True):
    # Read csv
    csv_data=pandas.read_csv(csv, names=['imgfn','lblfn','lbl','ignore'])

    ## Split by patient
    # Get IDs according to the imgfn column
    pats = np.array([get_id(x) for x in list(csv_data['imgfn'])])
    # Find Unique IDs
    patlist, patindices = np.unique(pats, return_inverse=True)

    if shuffle==True:
        np.random.shuffle(patlist)

    # Get train/val split of pats
    train_val_ind = int(len(patlist) * train_val_test[0])
    val_tst_ind =int(len(patlist) * (1-train_val_test[2]))

    # Get Indices
    train = patlist[:train_val_ind]
    val = patlist[train_val_ind:val_tst_ind]
    test = patlist[val_tst_ind:]

    # Define output filenames
    trainfn = join(out, 'train.csv')
    valfn = join(out, 'val.csv')
    testfn = join(out, 'test.csv')

    # Dump data to files
    pattocsv(trainfn, csv_data, train,pats)
    pattocsv(testfn, csv_data, test,pats)
    pattocsv(valfn, csv_data, val,pats)

def k_fold(csv, out,test=0.1, folds = 5, shuffle=True,class_balanced_test=True):
    # Read csv
    csv_data = pandas.read_csv(csv, names=['imgfn', 'lblfn', 'lbl'])

    ## Split by patient
    # Get IDs according to the imgfn column
    pats = np.array([get_id(x) for x in list(csv_data['imgfn'])])

    # Find Unique IDs
    patlist, patindices = np.unique(pats, return_inverse=True)

    np.random.shuffle(patlist)

    # Hold out a test set
    tst_ind =int(len(patlist) * (1-test))

    available = patlist[:tst_ind]
    kf  = KFold(n_splits=folds, shuffle=shuffle)
    kf.get_n_splits(available)
    
    # Split k-fold times
    split = 1
    for train_index,val_index in kf.split(available):
        train,val= available[train_index],available[val_index]
        trainfn = join(out, 'train_' + str(split) + '.csv')
        valfn = join(out, 'val_' + str(split) + '.csv')
        pattocsv(trainfn, csv_data, train, pats)
        pattocsv(valfn, csv_data, val, pats)
        split = split + 1

    test = patlist[tst_ind:]
    testfn = join(out, 'test.csv')
    pattocsv(testfn, csv_data, test, pats)