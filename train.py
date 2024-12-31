import argparse
import math
import os
from typing import TextIO
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from earlystop import EarlyStopping
from models import *
from dataset import LeafDataset
from evaltools import *


exp_number = 'res50'

# 命令行传参
# 跑st的时候bsize调成1
# st的bsize只能是1
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_of_epoch', type = int, default=1, help='num of epoch')
parser.add_argument('--batch_size', type = int, default=1, help='batch size')
parser.add_argument('--lr', type = float, default=1e-5, help='learning rate')
parser.add_argument('--train_label_dir', default='D:/Plant-Pathology-main/data/Train.csv', help='train label dir')
parser.add_argument('--train_image_dir', default='D:/Plant-Pathology-main/plant_dataset/train/images/', help='train image dir')
parser.add_argument('--val_label_dir', default='D:/Plant-Pathology-main/data/Val.csv', help='val label dir')
parser.add_argument('--val_image_dir', default='D:/Plant-Pathology-main/plant_dataset/val/images/', help='val image dir')
parser.add_argument('--save_dir', default=f'D:/Plant-Pathology-main/run/{exp_number}/', help='save dir')
parser.add_argument('--weights_dir', default='', help='weights path')
parser.add_argument("--log_dir_path", default=f'D:/Plant-Pathology-main/Log/', help='Log dir')
args = parser.parse_args()
early_stopping = EarlyStopping(save_path=args.save_dir, patience=6, verbose=True)
# 超参数

# 设置为gpu训练
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

num_of_epoch = args.num_of_epoch
learning_rate = args.lr
batch_size = args.batch_size

# 数据路径
train_label_dir = args.train_label_dir
train_image_dir = args.train_image_dir
val_label_dir = args.val_label_dir
val_image_dir = args.val_image_dir
save_dir = args.save_dir
weights_dir = args.weights_dir
log_dir = args.log_dir_path

# 创建结果目录
os.makedirs(save_dir, exist_ok=True)

# 打开log文件
train_val_log_f = open(log_dir + f"{exp_number}_train_val.log", 'w')


train_ds = LeafDataset(csv_file=train_label_dir, imgs_path=train_image_dir,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([312, 1000]),
                    torchvision.transforms.Normalize(mean=[0.40568268,0.51418954,0.3238448],std=[0.2018306,0.18794753,0.18870047],),
                    ]),
                    augment=True)

val_ds = LeafDataset(csv_file=val_label_dir, imgs_path=val_image_dir,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([312, 1000]),
                    torchvision.transforms.Normalize(mean=[0.40568268,0.51418954,0.3238448],std=[0.2018306,0.18794753,0.18870047],),
                    ]),
                    augment=True)

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4) 
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

TRAIN_SIZE = len(train_ds)
VALID_SIZE = len(val_ds)

loss_fn = torch.nn.BCEWithLogitsLoss()

print_running_loss = False

# 训练函数
def Train(net, loader):
    tr_loss = 0
    tr_accuracy = 0
    items = enumerate(loader)
    total_items = len(loader)
    all_labels = []
    all_predictions = []
    threshold = 0.5
    for _, (images, labels) in tqdm(items, total=total_items, desc="train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = net(images)  # 得到预测值
        loss = loss_fn(predictions, labels.squeeze(-1))
        net.zero_grad()
        loss.backward()
        tr_loss += loss.item()
        pred_d = (predictions.detach().cpu().numpy() >= threshold).astype(int)
        all_predictions.extend(pred_d)
        all_labels.extend(labels.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    tr_accuracy = Accuracy(y_true, y_pred)
    return tr_accuracy , tr_loss / total_items
# 验证函数
def Eval(net, loader):
    valid_loss = 0
    valid_accuracy = 0 
    all_labels = []
    all_predictions = []
    pred_all = []
    net.eval()
    net = net.cuda()
    threshold = 0.5
    with torch.no_grad():          
        # 创建一个迭代对象
        items = enumerate(loader)
        total_items = len(loader)  # 获取迭代对象的总长度    
        for _, (images, labels) in tqdm(items,total=total_items,desc="val"):       
            images, labels = images.to(device), labels.to(device)
            predictions = net(images)
            loss = loss_fn(predictions, labels.squeeze(-1))       
            valid_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            pred_0 = predictions.detach().cpu().numpy()
            pred_all.extend(pred_0)
            pred_d = (predictions.detach().cpu().numpy() >= threshold).astype(int)
            all_predictions.extend(pred_d)
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(pred_all)
    valid_accuracy = Accuracy(y_true, y_pred)
    overview(y_true, y_pred, y_prob, valid_loss/VALID_SIZE)
    return valid_accuracy, valid_loss/VALID_SIZE



# 训练
model_name = "ResNet50 with DE"
leaf_model = ResNet50(num_classes=6, pretrained=True).to(device)
# model_name = "VGG16 with DE"
# leaf_model = VGG16(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer with DE"
# leaf_model = SwinTransformer(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_STN_DE"
# leaf_model = SwinTransformer_STN(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_ResNet_ATT_DE"
# leaf_model = SwinTransformer_ResNet_ATT(num_classes=6, pretrained=True).to(device)
# model_name = "EfficientNet_DE"
# leaf_model = EfficientNet(num_classes=6, pretrained=True).to(device)
# model_name = "MobileNetV2 with data enhancement"
# leaf_model = MobileNetV2(num_classes=6, pretrained=True).to(device)
# model_name = "Xception"
# leaf_model = Xception(num_classes=6, pretrained=True).to(device)

# checkpoint continue to train
begin_epoch = 0
# leaf_model.load_state_dict(torch.load(f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/run/exp4/{begin_epoch}.pt'))

optimizer = optim.Adam(leaf_model.parameters(), lr=learning_rate)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
train_acc = []
val_acc = []

model_to_save = None # 将要保存的model
max_val_acc = -1

if __name__ == "__main__":
    # 训练
    for epoch in range(begin_epoch, num_of_epoch):
        print(f'Epoch {epoch+1}')
        leaf_model.train() # 设置模型的状态
        ta, tl = Train(leaf_model, loader=train_loader)
        train_acc.append(ta)
        train_loss.append(tl)
        print('Epoch: '+ str(epoch+1) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta))
        va, vl = Eval(leaf_model, loader=val_loader)
        valid_loss.append(vl)
        valid_acc.append(va)
        print('Epoch: '+ str(epoch+1) + ',  Val loss: ' + str(vl) + ', Val accuracy: ' + str(va))
        train_val_log_f.write('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta)
            + ', Val loss: ' + str(vl) + ', Val accuracy: ' + str(va) + '\n')
        if va >= max_val_acc:
            model_to_save = leaf_model  # 存一下当前的模型
            max_val_acc = va
            torch.save(model_to_save.state_dict(), save_dir + "bestvalvalacc.pt")
            print('bestvalvalacc.pt is saved successfully!')
        if epoch % 5 == 0:
            torch.save(leaf_model.state_dict(), save_dir + str(epoch) + ".pt")
            print(f'{str(epoch)}.pt is saved successfully!')

        early_stopping(vl, leaf_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 保存最终的模型
    torch.save(leaf_model.state_dict(), save_dir + "final.pt")
    print('final.pt is saved successfully!')
    
    # 画图
    epochs = range(1, len(train_loss) + 1) 
    plt.figure(1)
    plt.plot(epochs, train_loss, 'y', label='Training loss')
    plt.plot(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss' + ' Model: ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + "loss.jpg", dpi=800)

    plt.figure(2)
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Valid accuracy')
    plt.title('Training and validation accuracy' + ' Model: ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + "accuracy.jpg", dpi=800)