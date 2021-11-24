import argparse
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
from utils import *
from model import *
from sklearn.metrics import recall_score, f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--config', default = 'default', type =str) # Read UniTS hyperparameters
    parser.add_argument('--dataset', default = 'opportunity_lc', type = str,
                        choices=['opportunity_lc', 'seizure', 'wifi', 'keti'])
    parser.add_argument('--model', default='UniTS', type=str,
                        choices=['UniTS', 'THAT', 'RFNet', 'ResNet', 'MaDNN', 'MaCNN', 'LaxCat', 'static'])
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
    config = read_config(args.config + '.yaml')
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    args.log_path = os.path.join(args.log, args.dataset)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    torch.cuda.set_device(args.n_gpu)

    if args.dataset == 'opportunity_lc':
        args.input_size = 256
        args.input_channel = 45
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'seizure':
        args.input_channel = 18
        args.input_size = 256
        args.hheads = 6
        args.SENSOR_AXIS = 1
    elif args.dataset == 'wifi':
        args.input_channel = 180
        args.input_size = 256
        args.batch_size = 16
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'keti':
        args.input_channel = 4
        args.input_size = 256
        args.hheads = 4
        args.SENSOR_AXIS = 1
    args.model_save_path = os.path.join(args.log_path, args.model + '_'+ args.config + '.pt')
    return args, config

args, config = parse_args()
log = set_up_logging(args, config)
args.log = log


def test(model, xtest, ytest):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(xtest), args.batch_size):
            if i + args.batch_size <= len(xtest):
                x = torch.Tensor(xtest[i: i+args.batch_size]).cuda()
                y_true += ytest[i: i+args.batch_size]
            else:
                x = torch.Tensor(xtest[i:]).cuda()
                y_true += ytest[i:]  
            out = model(x)
            pred = torch.argmax(out, dim = -1)
            y_pred += pred.cpu().tolist()

    log("Accuracy : " + str(accuracy_score(y_true, y_pred)) +
        "\nMacro F1 : " + str(f1_score(y_true, y_pred, labels=list(range(args.num_labels)),average='macro')) )


def main():
    log("Start time:" + time.asctime( time.localtime(time.time())) )
    xtrain, ytrain, xtest, ytest = read_data(args, config)
    if args.model == 'UniTS':
        model = UniTS(input_size = args.input_size, sensor_num = args.input_channel, layer_num = config.layer_num,
        window_list = config.window_list, stride_list = config.stride_list, k_list = config.k_list,
        out_dim = args.num_labels, hidden_channel = config.hidden_channel).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    elif args.model == 'static':
        model = static_UniTS(input_size = args.input_size, sensor_num = args.input_channel, layer_num = config.layer_num,
        window_list = config.window_list, stride_list = config.stride_list, k_list = config.k_list,
        out_dim = args.num_labels, hidden_channel = config.hidden_channel).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    elif args.model == 'THAT':
        args.hlayers = 5
        args.vlayers = 1
        args.vheads = 16
        args.K = 10
        args.sample = 4
        model = HARTrans(args).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == 'RFNet':
        model = RFNet(num_classes = args.num_labels, input_channel = args.input_channel, win_len = args.input_size).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'ResNet':
        model = ResNet(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)        

    elif args.model == 'MaDNN':
        model = MaDNN(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   

    elif args.model == 'MaCNN':
        model = MaCNN(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels, 
            sensor_num = int(args.input_channel / args.SENSOR_AXIS)).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.model == 'LaxCat':
        model = LaxCat(input_size = args.input_size, input_channel = args.input_channel, num_label = args.num_labels,
            hidden_dim = 64, kernel_size = 32, stride = 8).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    loss_func = nn.CrossEntropyLoss()
    try:
        for ep in range(1, 1+args.epochs):
            model.train()
            epoch_loss = 0
            log("Training epoch : " + str(ep))
            for i in range(0, len(xtrain), args.batch_size):
                if i + args.batch_size <= len(xtrain):
                    x = torch.Tensor(xtrain[i: i+args.batch_size]).cuda()
                    y = torch.LongTensor(ytrain[i: i+args.batch_size]).cuda()  
                else:
                    x = torch.Tensor(xtrain[i:]).cuda()
                    y = torch.LongTensor(ytrain[i:]).cuda()                      
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
