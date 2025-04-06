import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import data_input
from sklearn import metrics
import matplotlib.pyplot as pl
import numpy as np
import timm
import time
import model1
from model2 import *
from model3 import *
from model4 import *
from model5 import *
from sklearn import metrics
import cv2
import os

# ferplus 

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

data_transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


def get_inputs(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = data_transform(image)
    inputs = torch.unsqueeze(inputs, dim=0)
    # print(type(input))
    return inputs


def predict(model, inputs):
    with torch.no_grad():
        # output = torch.squeeze(model(img))
        
        start_time = time.time()
        outputs = model(inputs)
       
        end_time = time.time()
        
        inference_time = end_time - start_time
        print(f"推理时间: {inference_time:.6f} 秒")
        probabilities = F.softmax(outputs, dim=1)
        index = probabilities.max(1).indices.item()
        value = probabilities.max(1).values.item()
        # print(index)

    return index, value


def main(model,img_path = './Confidence level/Transformer design.jpg'):#  Inkjet printing.png /Laster printing.png /Lithographic printing.png /Transformer design.jpg
    print(img_path)
    inputs = get_inputs(img_path)
    index, value= predict(model, inputs)
    text = 'Transformed design:' + f'{value:.4f}'
    output_folder = './Confidence level'
    output_filename = '4.png'
    add_text_to_image(img_path, text, output_folder, output_filename) 

    return index,class_indict[str(index)], value


# test_dir = 'G:/dataset/RAFDB/basic/Image/test/'

def pre_write_txt(pred, file):
    for i in pred:
        f = open(file, 'a', encoding='utf-8_sig')
        f.write(str(i) + ',')
        f.close()
    print("-----------------预测结果已经写入文本文件--------------------")

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None, save_path=None):
   
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
    pl.rcParams['font.family'] = 'Times New Roman'

    
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  

    
    if title is not None:
        pl.title(title, fontdict={'family': 'Times New Roman', 'size': 16})
   
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=0, fontfamily='Times New Roman')  
    pl.yticks(num_local, axis_labels, fontfamily='Times New Roman') 
    pl.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 14})
    pl.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 14})

    
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            value = cm[i][j] * 100
            pl.text(j, i, f"{value:.2f}%", ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontdict={'family': 'Times New Roman', 'size': 10}) 

    
    if save_path is not None:
        pl.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    
    pl.show()



validate_loader, val_num =  data_input.val_data(root_dir='../aug_xu_new/',batch_size=4)
def test_acc(model):
    acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pre=torch.tensor([])
    lab=torch.tensor([])
    with torch.no_grad():
        for data_test in validate_loader:
            
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)    
              
            outputs = model(test_images.to(device))
         
            predict_y = outputs.max(1, keepdim=True)[1]
             
            
            test_labels = test_labels.view_as(predict_y)
            # print(test_labels.shape)torch.Size([4, 1])
           
            predict_y=predict_y.to('cpu')                      
            test_labels=test_labels.to('cpu')
            pre=torch.cat([pre,predict_y],dim=0)
            lab=torch.cat([lab,test_labels],dim=0)
        
        accurate=metrics.accuracy_score(y_true=lab,y_pred=pre)   
        precition=metrics.precision_score(y_true=lab,y_pred=pre,average='macro')
        reacall=metrics.recall_score(y_true=lab,y_pred=pre,average='macro')
        f1=metrics.f1_score(y_true=lab,y_pred=pre,average='macro')

        plot_matrix(lab, pre, [0, 1, 2, 3], axis_labels=['0','1','2','3'] , save_path='CNN_Tran_parallel_confusion_matrix.png')#['Transformed design', 'Laster printing', 'Lithographic printing','Inkjet printing'])  title='confusion matrix',

    print(f'accurate: {accurate} precition: {precition} reacall: {reacall} f1: {f1}')
    total = sum([param.nelement() for param in model.parameters()])    
    print("\n模型的总参数量为: {:.2f}M".format(total/1e6))  



def class_acc():
    q = 0
    test_dir = 'G:/dataset/RAFDB/basic/Image/test/4/'
    imgs = os.listdir(test_dir)
    for img in imgs:
        img_path = test_dir + img
        index, true_label = main(img_path)
        if index == 3:
            q = q +1

    print("\n predict acc : {}".format(q/len(imgs)))

def add_text_to_image(image_path, text, output_folder, output_filename):
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image at path '{image_path}' was not found.")

    image = cv2.imread(image_path)
    
    # Calculate font scale based on image size
    font_scale = max(0.5, min(image.shape[0] / 500, 2))  # Scale font based on height, limits between 0.5 and 2

    # Define the text position and properties
    text_position = (image.shape[1] // 2 - 150, int(image.shape[0] * 0.1))  # Center horizontally, 10% from top
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 255)  # Red color in BGR
    thickness = 2

    # Add the text to the image
    cv2.putText(image, text, text_position, font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

    # Save the modified image
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)

    # Show the modified image
    cv2.imshow("Image with Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_model():
    model = CNN_Tran(img_size=128, patch_size=2, embed_dim=64, num_class=4)
    model_weight_path = "./savedmodel_CNN_Tran_entropy0.864.pth"

    # model = CNN_Tran_parallel(img_size=128, patch_size=2, embed_dim=64, num_class=4)
    # model_weight_path = "./savedmodel_parallel0.845.pth"

    # model = model1.ResNet18()
    # model_weight_path = "./savedmodel_ResNet0.637.pth"

    # model = transformer(img_size=128, patch_size=2, embed_dim=64, num_class=4)
    # model_weight_path = "./savedmodel_transformer0.455.pth"

    # model = model1.CNN(img_size=128, patch_size=2, embed_dim=64, num_class=4)
    # model_weight_path = "savedmodel_CNN0.83.pth"
    
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    return model

if __name__ == '__main__':
    
    model = get_model()
   
    main(model)
   
    # test_acc(model)

