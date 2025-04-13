import matplotlib.pyplot as plt
import numpy as np

# # Read trainset_acc data from file
# with open('./result_woattn_CNN.txt', 'r') as file:
#     lines = file.readlines()
#     accuracy_model_1 = [float(line.split('trainset_acc:')[1].split()[0]) for line in lines if 'trainset_acc' in line]
# # print(accuracy_model_1[1])
# # Extract epochs based on the length of the accuracy data
# epochs = np.arange(1, len(accuracy_model_1) + 1)

# # Sample data for demonstration (you can replace these with actual data from other models if available)
# with open('./result_VisionTran.txt', 'r') as file:
#     lines = file.readlines()
#     accuracy_model_2 = [float(line.split('trainset_acc:')[1].split()[0]) for line in lines if 'trainset_acc' in line]
# with open('./result_mobilenet.txt', 'r') as file:
#     lines = file.readlines()
#     accuracy_model_3 = [float(line.split('trainset_acc:')[1].split()[0]) for line in lines if 'trainset_acc' in line]
# with open('./result_vgg.txt', 'r') as file:
#     lines = file.readlines()
#     accuracy_model_4 = [float(line.split('trainset_acc:')[1].split()[0]) for line in lines if 'trainset_acc' in line]
# with open('./result_woattn_CNN_Trans_entropy0.864.txt', 'r') as file:
#     lines = file.readlines()
#     accuracy_model_5 = [float(line.split('trainset_acc:')[1].split()[0]) for line in lines if 'trainset_acc' in line]

# Read trainset_acc data from file
with open('./result_woattn_CNN.txt', 'r') as file:
    lines = file.readlines()
    accuracy_model_1 = [float(line.split('train_loss:')[1].split()[0]) for line in lines if 'train_loss' in line]
# print(accuracy_model_1[1])
# Extract epochs based on the length of the accuracy data
epochs = np.arange(1, len(accuracy_model_1) + 1)

# Sample data for demonstration (you can replace these with actual data from other models if available)
with open('./result_VisionTran.txt', 'r') as file:
    lines = file.readlines()
    accuracy_model_2 = [float(line.split('train_loss:')[1].split()[0]) for line in lines if 'train_loss' in line]
with open('./result_mobilenet.txt', 'r') as file:
    lines = file.readlines()
    accuracy_model_3 = [float(line.split('train_loss:')[1].split()[0]) for line in lines if 'train_loss' in line]
with open('./result_vgg.txt', 'r') as file:
    lines = file.readlines()
    accuracy_model_4 = [float(line.split('train_loss:')[1].split()[0]) for line in lines if 'train_loss' in line]
with open('./result_woattn_CNN_Trans_entropy0.864.txt', 'r') as file:
    lines = file.readlines()
    accuracy_model_5 = [float(line.split('train_loss:')[1].split()[0]) for line in lines if 'train_loss' in line]

# Plotting the accuracies
plt.figure(figsize=(10, 5))
# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(epochs, accuracy_model_1, color='purple', linewidth=2, label='ResNet18')
plt.plot(epochs, accuracy_model_2, color='red', linewidth=2, label='Vision Transformer')
plt.plot(epochs, accuracy_model_3, color='cyan', linewidth=2, label='MobileNet V2')
plt.plot(epochs, accuracy_model_4, color='orange', linewidth=2, label='VGG16')
plt.plot(epochs, accuracy_model_5, color='green', linewidth=2, label='Ours')

# Customizing the plot
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 2)  
plt.grid(True)
plt.legend(fontsize=12)

# Save and display the plot
plt.tight_layout()
plt.savefig('loss_vs_epoch_plot.png', bbox_inches='tight', dpi=300)
plt.show()
