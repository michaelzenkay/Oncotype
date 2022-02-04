import sys
from ClassNetTrainer import ClassNetTrainer

#--------------------------------------------------------------------------------
def main(csv='./infer.csv',
         csv_out='./inferred.csv', 
         classes = 2, 
         batch = 50,
         resize = 256,
         crop = 224,
         modelfn='./model.pth'):
    arch = 'res'
    gpus = [1]

### Inference
    network = ClassNetTrainer()
    network.inference(csv, arch, classes, batch, resize, crop, modelfn, csv_out,gpu=gpus)

#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    if len (sys.argv) > 1:    
        print(sys.argv[1] + ' -input csv')
        print(sys.argv[2] + ' -output csv')
        if len(sys.argv) > 3:
            print(sys.argv[3] + ' -modelfn')
            main(csv = sys.argv[1], csv_out = sys.argv[2], modelfn = sys.argv[3])
        else:
            main(csv = sys.argv[1], csv_out = sys.argv[2])
    else:
        main()