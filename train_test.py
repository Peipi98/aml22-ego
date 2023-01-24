import os
import pickle as pk
import numpy as np
import torch
from classifier_test import Classifier
import time
from datetime import datetime
from statistics import mean
from utils.utils import AverageMeter, Accuracy 
from utils.logger import logger
import torch.nn.parallel
import torch.optim


SAMPLING = 'dense'
TYPE = 'train'
N_FRAMES = 5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_classifier = Classifier(1024, 512, 8)
    action_classifier = load_on_gpu(action_classifier, device)
    optimizer = torch.optim.SGD(action_classifier.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = dataloader('train')

    val_loader = dataloader('test')

    train(train_loader, optimizer, action_classifier, criterion , 30)
    
    return

def train(dataloader, optimizer, classifier, criterion, num_epochs):
    for epoch in range(num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        #switch to training mode
        classifier.train()

        end = time.time()
        for i, (input, target) in enumerate(dataloader):
            data_time.update(time.time() - end)

            # target = target.cuda(async=True)
            input = input.reshape((1, input.shape[0], input.shape[1]))
            
            print(input)
            print(input.shape)
            print(target)
            #print(target.shape)
            input_var = torch.Tensor(input)
            target_var = torch.Tensor(np.array(target))
            print(input_var)
            print(input_var.shape)
            print(target_var)
            print(target_var.shape)
            
            # compute output
            output = classifier(input_var)
            loss = criterion(output, target_var)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(dataloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return 

def eval(classifier, val_dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for input, target in val_dataloader:
            output = classifier(input)
            _, predicted = torch.max(output.data,1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(accuracy)
        

def dataloader(type):
    _path = f'./saved_features/{SAMPLING}/{type}/'
    file_name = f'save_feat_I3D_D1_EK_{type}_{SAMPLING}_{N_FRAMES}.pkl'

    features = []
    
    for file in os.listdir(_path):
        if f"{_path}{file_name}" == os.path.join(_path, file):
            path = os.path.join(_path, file)
            tmp = open(path, 'rb')
            pk_file = pk.load(tmp)
            tmp.close()

            tmp_arr = []
            for feat in pk_file['features']:
                tmp_arr.append(feat['features_RGB'])
            features = np.array(tmp_arr)
            
    tmp = open(f'./train_val/D1_{type}.pkl', 'rb')
    pk_file = pk.load(tmp)
    tmp.close()
    labels = []
    for i,line in pk_file.iterrows():
        labels.append(line['verb_class'])
        

    # Convert features to numpy array
    _dataloader = list(zip(features, labels))
    return _dataloader
    
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_on_gpu(model, device=torch.device('cuda')):
    """
    function to load the classifier related to the task on the different GPUs used
    """
    return torch.nn.DataParallel(model).to(device)

if __name__ == '__main__':
    main()

