import argparse
import os
import time
import random
import torch
import numpy as np
from utils import *
from model.STF import *
from sklearn.metrics import recall_score, f1_score, accuracy_score


def parse_args():
    '''
     Read exp args and model config
    '''
    parser = argparse.ArgumentParser(description='train and test for complex model')
    parser.add_argument('--dataset', default = 'opportunity_lc', type = str)
    parser.add_argument('--model', default='STF', type=str,
                        choices=['STF'])
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--log', default='log', type=str,
                        help="Log directory")
    parser.add_argument('--exp', default='', type=str,
                        choices = ['','noise','missing_data'])
    parser.add_argument('--ratio', default=0.2, type=float)
    parser.add_argument('--n_gpu', default=0, type =int)
    
    parser.add_argument('--epochs', default = 50, type = int)
    parser.add_argument('--lr', default = 1e-3, type = float)
    parser.add_argument('--batch_size', default = 64, type = int)

    parser.add_argument('--save', action = 'store_true')
    parser.add_argument('--test_only', action = 'store_true')
    args = parser.parse_args()
    args.config = 'none'
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    args.log_path = os.path.join(args.log, args.dataset)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    torch.cuda.set_device(args.n_gpu)

    if args.dataset == 'opportunity_lc':
        args.input_size = 256
        args.input_channel = 45
        args.sensor_type = 3
        args.SENSOR_AXIS = 3
        args.hheads = 9
    elif args.dataset == 'hhar':
        args.input_size = 512
        args.input_channel = 6
        args.sensor_type = 2
        args.SENSOR_AXIS = 3
        args.hheads = 2
        args.sensor_type = 2
    elif args.dataset == 'seizure':
        args.input_channel = 18
        args.input_size = 256
        args.SENSOR_AXIS = 1
        args.hheads = 6
        args.sensor_type = 1
    elif args.dataset == 'wifi':
        args.input_channel = 180
        args.input_size = 256
        args.batch_size = 16
        args.SENSOR_AXIS = 3
        args.hheads = 9
        args.sensor_type = 2
    args.model_save_path = os.path.join(args.log_path, args.model + '_'+ args.config + '.pt')
    return args, {}

args, config = parse_args()
log = set_up_logging(args, config)
args.log = log

def test(model, xtest, ytest):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(xtest) - args.batch_size, args.batch_size):
            x = torch.Tensor(xtest[i: i+args.batch_size]).cuda()
            y_true += ytest[i: i+args.batch_size]  
            out = model(x)
            pred = torch.argmax(out, dim = -1)
            y_pred += pred.cpu().tolist()

    log("Accuracy : " + str(accuracy_score(y_true, y_pred)) +
        "\nMacro F1 : " + str(f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')) )


def main():
    log("Start time:" + time.asctime( time.localtime(time.time())) )
    xtrain, ytrain, xtest, ytest = read_data(args, config)

    if args.model == 'STF':
        args.FILTER_LEN = [3, 3, 3, 3]
        args.DILATION_LEN = [1, 2, 4, 8]
        if args.input_size == 512:
            args.GEN_FFT_N2 = [12, 24, 48, 96]
            #SERIES_SIZE2 = 384
            args.GEN_FFT_N = [16, 32, 64, 128]
            args.GEN_FFT_STEP2 = [FFT_N_ELEM for FFT_N_ELEM in args.GEN_FFT_N2]
            args.GEN_FFT_STEP = [FFT_N_ELEM for FFT_N_ELEM in args.GEN_FFT_N]
        elif args.input_size == 256:
            args.GEN_FFT_N2 = [6, 12, 24, 48]
            #SERIES_SIZE2 = 384
            args.GEN_FFT_N = [8, 16, 32, 64]
            args.GEN_FFT_STEP2 = [FFT_N_ELEM for FFT_N_ELEM in args.GEN_FFT_N2]
            
            args.GEN_FFT_STEP = [FFT_N_ELEM for FFT_N_ELEM in args.GEN_FFT_N]

        args.GEN_C_OUT = 72
        model = STFNet(args).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


    total_params = sum(p.numel() for p in model.parameters())
    log('Total parameters: ' + str(total_params))

    if args.test_only:
        if os.path.exists(args.model_save_path):
            model.load_state_dict(torch.load(args.model_save_path))
            test(model, xtest, ytest)
        else:
            log("Model state dict not found!")
        return

    random.seed(args.seed)
    random.shuffle(xtrain)
    random.seed(args.seed)
    random.shuffle(ytrain)

    loss_func = torch.nn.CrossEntropyLoss()
    try:
        for ep in range(1, 1+args.epochs):
            model.train()
            epoch_loss = 0
            log("Training epoch : " + str(ep))
            for i in range(0, len(xtrain)-args.batch_size, args.batch_size):
                x = torch.Tensor(xtrain[i: i+args.batch_size]).cuda()
                y = torch.LongTensor(ytrain[i: i+args.batch_size]).cuda()                      
                out = model(x)
                loss = loss_func(out, y)
                epoch_loss += loss.cpu().item()

                optimizer.zero_grad()           
                loss.backward()
                optimizer.step()

            log("Training loss : " + str(epoch_loss / (i / args.batch_size + 1)))
            test(model, xtest, ytest)
            log("----------------------------")


    except KeyboardInterrupt:
        print('Exiting from training early')
        test(model, xtest, ytest)
    if args.save:
        torch.save(model.state_dict(), args.model_save_path)

if __name__ == '__main__':
    main()
