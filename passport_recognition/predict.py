import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import json,os
from model7 import *

try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

data_transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu' 

# read class_indict

def get_model():
    # create model
    model = cnn_tran(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)
    # load model weights
    model_weight_path = "./savedmodel_woattn0.837.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model
    
# 单张预测
def predict_one(model,img_path = "../tulips.jpg"):
    # load image
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)  # 预处理的时候已经将channel这个维度提到最前面
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 在最前面增加一个batch维度
    with torch.no_grad():
        # predict class
        # model.to(device)
        output = torch.squeeze(model(img))  # 将batch维度压缩掉
        # print(output.shape)
        predict = torch.softmax(output, dim=0)  # 变成概率分布
        predict_cla = int(torch.argmax(predict).numpy())  # 获得最大概率处的索引值
    
    # print(class_indict[str(predict_cla)], predict[predict_cla].item())
    return predict_cla,class_indict[str(predict_cla)]

# model = get_model()
# index,true_lable = predict(model)
# print(index,true_lable)
# 以下都是用单张的方式预测整个测试集
def get_inputs(img_path):
    image = Image.open(img_path)
    inputs = data_transform(image)
    inputs = inputs.unsqueeze(0)
    return inputs

def predict(model, inputs):
    with torch.no_grad():
         # output = torch.squeeze(model(img))
        outputs = model(inputs)
        index = outputs.max(1).indices.item()
        # print(index)

    return index

# 预测一张图像
def main(model,img_path):
    inputs = get_inputs(img_path)
    index= predict(model, inputs)
    return index

label_dict = {'train_01bianxing': '0', 'train_02jiguang': '1','train_03jiaoyin': '2','train_04penmo': '3'}
img_list = []
label_list = []
def input_data(test_dir):
    for dir in os.listdir(test_dir):
        imgs_dir = test_dir + dir + '/'
        print(imgs_dir)
        for img_name in os.listdir(imgs_dir):
            img_path = imgs_dir + img_name
            img_list.append(img_path)
            label_list.append(int(label_dict[dir]))
    # print(img_list)
    return img_list,label_list

# def get_model():
#     model = AlexNet(num_classes=5)
#     # load model weights
#     model_weight_path = "./AlexNet.pth"
#     model.load_state_dict(torch.load(model_weight_path))
#     model.eval()
#     return model

#通过一张张图像预测（单独读取）
def predice_testset(model):
    n = 0
    imgs, true_labels = input_data(test_dir='../aug_xu_new/test/')
    # print(true_labels)
    for i in range(len(imgs)):
        # print(imgs[i])
        result_indx = main(model,imgs[i])
        # print("-------------------result_indx:{}".format(result_indx))
        if result_indx == int(true_labels[i]):
            # print('n')
            n = n +1
    print("\n predict acc : {}".format(n/len(imgs)))

# input_data(test_dir='../data/val/')
model = get_model()
predice_testset(model)


