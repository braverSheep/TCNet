## TCNet: Hybrid CNN-Transformer Framework with Dynamic Feature Fusion for Enhanced Passport Background Texture Classification
### Maoqin Tian, Lin Tang, JiaFeng Xu, Yibo Zhang, Yong Yang, Lingpei Zeng, Eryang Chen, Yuanlun Xie
---
This work is being submitted to the journal “The Visual Computer” for publication
===
Verification of passport authenticity is particularly important for global security and border management. The recognition of passport background textures is one of the key technologies for determining the authenticity of passports. This work proposes a complementary CNN and Transformer framework, namely TCNet, for the recognition of passport background textures. The framework uses CNN to extract local features of the background texture of the passport, uses Transformers to model the global features of the texture, and finally uses a dynamic weighting module (DWM) to fuse the features of the two, ultimately improving the performance of classifying the background texture of the passport.



## Pytorch
We used Pytorch for the implementation of our code.

## Requirements
1. Python 3.11
2. torch 2.3.1+cu121
3. torchvision 0.18.1+cu121
4. cuda 12.1

Only a basic environment setup is needed to implement our method.

## 📁 Folder structure

Download all files from our repo. The following shows the basic folder structure.

```text

├── Confidence level
├── Grad # testing code
├── Model parameters # training code
│   ├── savedmodel_CNN_Tran_entropy0.864.pth # ours
│   └── savedmodel_CNN0.83.pth               # ablation_CNN
|   └── savedmodel_parallel0.845.pth         # ablation_CNN_Transformer no DWM
|   └── savedmodel_transformer0.455.pth      # ablation_Transformer
├── Pictures               
├── training_record
├── bubble.py              # Generate bubble chart
├── data_input.py          # the code for data processing
├── model5.py              # the proposed model
├── test_single.py         # testing code
├── train.py               # training code
├── visual_heatmap.py      # the code for Grad-CAM visualization

```


## Dataset
A passport backing dataset with four main types of backing: 'Transformed design', 'Laser printing', 'Lithographic printing' and 'Inkjet printing', with labels assigned as 0, 1, 2, and 3, respectively, is developed . In particular, 'Transformed design' and 'lithographic printing' are the shading expressions of real passports, and the remaining two are the shading expressions of fake passports. In our study, the dataset was split into a training set and a test set in an 8:2 ratio. <br>
To enhance the diversity and richness of the dataset, data augmentation techniques were applied, improving the model's generalization ability. We augmented the captured images with random horizontal flips, vertical flips, random rotations, color dithering, random affine transformations, and perspective transformations, which increased the number of samples per category in the training set to 1000, and 250 samples per category in the test set. The dataset can be obtained from website :[passport backing dataset](https://pan.baidu.com/s/1pBBza-5w_56DTDZlO3sATQ?pwd=ufxt) <br>

If you wish to use this dataset, please cite the following:<br>
@article{xu2024pbnet,<br>
  title={PBNet: Combining Transformer and CNN in Passport Background Texture Printing Image Classification},<br>
  author={Xu, Jiafeng and Jia, Dawei and Lin, Zhizhe and Zhou, Teng and Wu, Jie and Tang, Lin},<br>
  journal={Electronics},<br>
  volume={13},<br>
  number={21},<br>
  pages={4160},<br>
  year={2024},<br>
  publisher={MDPI}<br>
}

## Train our model
```text
python train.py  
```

## Test in test dataset
```text
python test_single.py
```

## Contact
If you have any questions, please contact Yuanlun Xie at:[fengyuxiexie@163.com](fengyuxiexie@163.com)
