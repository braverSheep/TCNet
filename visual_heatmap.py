
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import timm
from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import model1
from model2 import *
from model3 import *
from model4 import *
from model5 import *


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load the image
image_path = "./Grad/Inkjet print.png" #Laster printing.png   Lithograpich printing.png  Transformed design.jpg  Inkjet print.png
image_name = image_path.split('/')[-1].split('.')[0]
superimposed_image_path = './Grad/{}_CNN_transformer.jpg'.format(image_name)
data_transform = {
    "train": transforms.Compose([
                                 transforms.Resize((128, 128)),
                                 # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                 # transforms.RandomResizedCrop(128), 
                                 transforms.ColorJitter(hue=0.5),
                                 # transforms.RandomRotation((-45,45)),
                                 transforms.RandomHorizontalFlip(), 
                                 # transforms.RandomRotation((-45, 45)),                                                            
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                 ]),

    "val": transforms.Compose([transforms.Resize((128, 128)),
                               # transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])}

image = Image.open(image_path).convert("RGB")
inputs = data_transform['val'](image)
inputs = torch.unsqueeze(inputs, dim=0).to(device)

# load the trained model

# model = CNN_Tran(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)
# model = model1.CNN(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)
# model = transformer(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)
model = CNN_Tran_parallel(img_size=128, patch_size=2, embed_dim=64, num_class=4).to(device)

# load model weights
# model_weight_path = "./savedmodel_CNN_Tran_entropy0.864.pth"
# model_weight_path = "./savedmodel_CNN0.83.pth"
# model_weight_path = "./savedmodel_transformer0.455.pth"
model_weight_path = "./savedmodel_parallel0.845.pth"

model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cuda:0')))
# print("---------------------------------")
# model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu'))['model_state_dict'])
# print(model)

# # forward
model.eval()
outputs, feature_maps = model(inputs)
features = feature_maps

pred_label = torch.argmax(outputs).item()
pred_score = outputs[:, pred_label]
features_grad = None

# # trans visualizer
# use_cuda = torch.cuda.is_available()
# cam = GradCAM(model=model,
#             target_layers=[model.blocks[-1].norm1],
#             use_cuda=use_cuda,
#             reshape_transform=reshape_transform)

# cam = GradCAM(model=model,
#             target_layers=[model.blocks[5].norm1],
#             use_cuda=use_cuda
#            )

# target_category = None 
# grayscale_cam = cam(input_tensor=inputs, targets=target_category)
# features = grayscale_cam[0, :]
# print(features)

# # image_rgb = image.resize((224,224))
# # image_rgb = np.array(image_rgb)
# # def normalization(data):
# #     _range = np.max(data) - np.min(data)
# #     return (data - np.min(data)) / _range

# # image_rgb = normalization(image_rgb)
# # visualization = show_cam_on_image(image_rgb, grayscale_cam)


# image = cv2.imread(image_path)
# visualization = cv2.resize(features, (image.shape[1], image.shape[0]))
# visualization = np.uint8(255 * visualization)
# # visualization = visualization.astype(np.uint8)
# heatmap = cv2.applyColorMap(np.uint8(255 * visualization), cv2.COLORMAP_JET)
# print(image)
# superimposed_image = heatmap * 0.2 + image  # 0.4 is the intensity factor of heatmap
# print(superimposed_image)
# cv2.imwrite(superimposed_image_path, superimposed_image)



# a auxiliary function that help to read intermediate gradient
def extract(g):
    global features_grad
    features_grad = g


features.register_hook(extract)
pred_score.backward()

heatmap = features.detach().cpu().numpy()
heatmap = heatmap[0,:,:,:]
heatmap = np.mean(heatmap, axis=0)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(heatmap.shape)

grads = features_grad

pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
pooled_grads = pooled_grads[0]
print(pooled_grads.size()[0])
features = features[0]

for i in range(pooled_grads.size()[0]):
    features[i, ...] *= pooled_grads[i, ...]



image = cv2.imread(image_path)


heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)


heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


superimposed_image = heatmap * 0.4 + image


cv2.imwrite(superimposed_image_path, superimposed_image)


# cv2.imshow("heatmap", superimposed_image)
# cv2.waitKey(0)
