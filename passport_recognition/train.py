import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import  torchvision.models as models 
#import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import sys
import data_input
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import timm
from tqdm import *

import os
import json
import time

import model1
from model2 import *
from model3 import *
from model4 import *
from model5 import *

from sklearn import metrics



def seed_torch(seed=3407):
    import random
    # seed = random.randint(1, 20000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
seed_torch()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = '../aug_xu_new/'
batch_size = 10
lr_gamma = 0.8
# trainset loader
train_loader, tra_num,imgs = data_input.train_data(data_dir,batch_size)
print(len(train_loader))
# validation loader
validate_loader, val_num = data_input.val_data(data_dir,2)

net = CNN_Tran(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)
print('CNN')

pre_train_model = 'AlexNet.pth'
save_path = './AlexNet.pth'

writer=SummaryWriter()

# loss_function = nn.CrossEntropyLoss()
loss_function = FocalLoss()

pata = list(net.parameters())  

#optimizer = optim.Adam(net.parameters(), lr=0.0001)
optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)##0.0003
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=0, last_epoch=-1)

is_recovery = False
if is_recovery:
    checkpoint = torch.load(pre_train_model)
    net.load_state_dict(checkpoint)
    # optim.Adam(net.parameters(), lr=0.05).load_state_dict(checkpoint["optimizer_state_dict"])
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr
    #     start_epoch = checkpoint["epoch"]
    #     best_accuracy = checkpoint["best_accuracy"]
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.3, last_epoch=start_epoch)
    print("-------- 模型恢复训练成功-----------")
    save_path = './pretrain_{}_AlexNet.pth'.format(pre_train_model.split('.')[0])


def pre_write_txt(pred, file):
    f = open(file, 'a', encoding='utf-8')
    f.write(str(pred))
    f.write('\n')
    f.close()
    print("-----------------预测结果已经写入文本文件--------------------")


best_acc = 0.0
best_f1=0.0

for epoch in tqdm(range(100)):
   
    # train
    net.train()  
    running_loss = 0.0
    t1 = time.perf_counter() 
    # print('star')
    tra_acc = 0.0
    for step,data in enumerate(train_loader, start=0):
        # print(step)

       
        images, labels = data
        # print(labels)
        optimizer.zero_grad()
        outputs = net(images.to(device))
        # soft_outputs = torch.softmax(outputs, dim=1)
        # print(soft_outputs)
        # print(soft_outputs.max(1),torch.argmax(soft_outputs))
        loss = loss_function(outputs, labels.to(device))
        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        tra_predict_y = torch.max(outputs, dim=1)[1]
        step_acc = (tra_predict_y == labels.to(device)).sum().item()
        tra_acc += step_acc
        running_loss += loss.item()
        # each 10 step(or batch) print once
        # if (step+1)%10 == 0:
         # print("step:{} train acc:{:.3f} train loss:{:.3f}".format(step,step_acc/len(labels),loss))
    one_epoch_time = time.perf_counter()-t1
    writer.add_scalar('acc', tra_acc / tra_num, epoch)
    writer.add_scalar('loss', running_loss / step, epoch)
    # validate
    net.eval()  
    acc = 0.0  # accumulate accurate number / epoch
    pre=torch.tensor([])
    lab=torch.tensor([])
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)
            outputs = net(test_images.to(device))
            outputs=outputs.to('cpu')                      
            test_labels=test_labels.to('cpu')
            _,predict_y = torch.max(outputs, dim=1)       
            # acc += (predict_y == test_labels.to(device)).sum().item()
            pre=torch.cat([pre,predict_y],dim=0)
            lab=torch.cat([lab,test_labels],dim=0)

        # accurate_test = acc / val_num
        accurate_test=metrics.accuracy_score(y_true=lab,y_pred=pre)  
        precition_test=metrics.precision_score(y_true=lab,y_pred=pre,average='macro')
        reacall_test=metrics.recall_score(y_true=lab,y_pred=pre,average='macro')
        f1_test=metrics.f1_score(y_true=lab,y_pred=pre,average='macro')
        if accurate_test > best_acc and f1_test > best_f1 :
            best_acc = accurate_test
            best_f1=f1_test
            torch.save(net.state_dict(), f'./savedmodel_half_feature{best_acc}.pth')
            print("The model was saved with accurate_test : {} f1_test:{} ".format(accurate_test,f1_test))

        print('\n[epoch %d] trainset_acc:%.4f train_loss: %.4f  testset_accuracy: %.4f test_precition: %.4f testset_recall: %.4f testset_f1: %.4f  best_acc: %.4f one_epoch_time:%.3fs\n' %
              (epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,precition_test,reacall_test,f1_test,best_acc,one_epoch_time))
        pre_write_txt("epoch:{} trainset_acc:{:.4f} train_loss:{:.4f} testset_accuracy: {:.4f} test_precition: {:.4f} testset_recall: {:.4f} testset_f1: {:.4f} best_acc: {:.4f} best_f1: {:.4f}".format(epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,precition_test,reacall_test,f1_test,best_acc,best_f1), file = 'result_woattn.txt')
    lr_scheduler.step()
writer.close()


print('Finished Training')
