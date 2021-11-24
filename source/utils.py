import os
import csv
import yaml
import numpy as np

def read_data(args, config):
    path = os.path.join('../dataset', args.dataset)
    x_train = np.load(os.path.join(path, 'x_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy')).astype('int64').tolist()
    x_test = np.load(os.path.join(path, 'x_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy')).astype('int64').tolist()
    np.random.seed(args.seed)

    if args.exp == 'noise': # Robustness test (noise)
        for i in range(len(x_train)):
            for j in range(x_train.shape[2]):
                noise = np.random.normal(1,1 , size= x_train[i][:, j].shape)
                x_train[i][:, j] = x_train[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_train[i][:, j] ))
        for i in range(len(x_test)):
            for j in range(x_test.shape[2]):
                noise = np.random.normal(1, 1, size= x_test[i][:, j].shape)
                x_test[i][:, j] = x_test[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_test[i][:, j] ))

    elif args.exp == 'missing_data': # Robustness test (missing value)
        for i in range(len(x_train)):
            for j in range(x_train.shape[2]):
                mask = np.random.random(x_train[i][:, j].shape) >= args.ratio
                x_train[i][:, j] = x_train[i][:, j] * mask
        for i in range(len(x_test)):
            for j in range(x_test.shape[2]):
                mask = np.random.random(x_test[i][:, j].shape) >= args.ratio
                x_test[i][:, j] = x_test[i][:, j] * mask

    args.num_labels = max(y_train) + 1
    summary = [0 for i in range(args.num_labels)]
    for i in y_train:
        summary[i] += 1
    args.log("Label num cnt: "+ str(summary))
    args.log("Training size: " + str(len(y_train)))
    args.log("Testing size: " + str(len(y_test)))
    return list(x_train), y_train, list(x_test), y_test

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')
    return write_log

def set_up_logging(args, config):
    log = logging(os.path.join(args.log_path, args.model+'.txt'))
    for k, v in config.items():
        log("%s:\t%s\n" % (str(k), str(v)))
    return log

