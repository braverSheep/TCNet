import torch
from torchvision import transforms, datasets, utils
import json
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True 
# transforms.transforms.RandomErasing(p=0.3, scale=(0.08, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)


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


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
   
def train_data(root_dir,batch_size):
    train_dataset = datasets.ImageFolder(root=root_dir + "train",transform=data_transform["train"])
    train_data_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx  
    cla_dict = dict((val, key) for key, val in flower_list.items()) 
    
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file: 
        json_file.write(json_str)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=0)
    return train_loader,train_data_num,train_dataset.imgs
def val_data(root_dir,batch_size):
    validate_dataset = datasets.ImageFolder(root=root_dir + "test",transform=data_transform["val"])
    val_data_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=0)
    return validate_loader,val_data_num

if __name__== "__main__":
    data_dir = '../aug_xu_new/'
    batch_size = 32
    validate_loader,val_data_num, _ = train_data(data_dir,batch_size)
    print(val_data_num)
    # val_num = 0
    # for data_test in validate_loader:
    #         test_images, test_labels = data_test
    #         test_labels_len = len(test_labels)
    #         val_num = val_num + test_labels_len
    #         print(test_labels_len)
 
