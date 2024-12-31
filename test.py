import argparse
import math
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import ttach as tta
import numpy as np
from models import *
from dataset import LeafDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,hamming_loss
from evaltools import *


exp_number = 'res50'
ttaflag = None

# 测试的batchsize一定要是1
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_of_epoch', type = int, default=1, help='num of epoch')
parser.add_argument('--batch_size', type = int, default=1, help='batch size')
parser.add_argument('--lr', type = float, default=1e-5, help='learning rate')
parser.add_argument('--test_label_dir', default='D:/Plant-Pathology-main/data/Test.csv', help='test label dir')
parser.add_argument('--test_image_dir', default='D:/Plant-Pathology-main/plant_dataset/test/images/', help='test image dir')
parser.add_argument('--save_dir', default=f'D:/Plant-Pathology-main/test/{exp_number}/', help='save dir')
parser.add_argument('--weights_dir', default='', help='weights path')
args = parser.parse_args()

# 参数传递
num_of_epoch = args.num_of_epoch
learning_rate = args.lr
batch_size = args.batch_size

# 数据路径
test_label_dir = args.test_label_dir
test_image_dir = args.test_image_dir
save_dir = args.save_dir
weights_dir = args.weights_dir

# 创建结果目录
# os.system("mkdir -p " + save_dir)


# 创建Loader
# st记得改成batchsize=1 [384, 384], 其余resize[312, 1000]
test_ds = LeafDataset(csv_file=test_label_dir, imgs_path=test_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([312,1000]),
                    torchvision.transforms.Normalize(mean=[0.40568268,0.51418954,0.3238448],std=[0.2018306,0.18794753,0.18870047],),
                    ]),
                    augment=None)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.BCEWithLogitsLoss()

TEST_SIZE = len(test_ds)

# 读取模型
model_save_root_path = f'D:/Plant-Pathology-main/run/{exp_number}/best.pt'

model_name = "ResNet50_DE"
leaf_model = ResNet50(num_classes=6, pretrained=True)
# model_name = "VGG16_DE"
# leaf_model = VGG16(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_DE"
# leaf_model = SwinTransformer(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_STN_DE"
# leaf_model = SwinTransformer_STN(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_ResNet_ATT"
# leaf_model = SwinTransformer_ResNet_ATT(num_classes=6, pretrained=True).to(device)
# model_name = "EfficientNet_DE"
# leaf_model = EfficientNet(num_classes=6, pretrained=True).to(device)
# model_name = "MobileNetV2_DE"
# leaf_model = MobileNetV2(num_classes=6, pretrained=True).to(device)
# model_name = "Xception"
# leaf_model = Xception(num_classes=6, pretrained=True).to(device)
leaf_model.load_state_dict(torch.load(model_save_root_path, weights_only=True))
leaf_model.eval()

# 打开存储的Log文件
path = 'D:/Plant-Pathology-main/test/test.log'
fa = open(path, 'a')

csv_path_txt = f'D:/Plant-Pathology-main/test/{exp_number}.txt'
f = open(csv_path_txt, 'w')
f.write('healthy,scab,frog_eye_leaf_spot,rust,complex,powdery_mildew,healthy,scab,frog_eye_leaf_spot,rust,complex,powdery_mildew\n')


# defined 2 * 2 * 3 * 3 = 36 augmentations !
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)
tta_model = tta.ClassificationTTAWrapper(leaf_model, transforms)
if(ttaflag == True):
    leaf_model = tta_model


def print_log(pred, label):
    for i in range(0, len(pred)):
        for j in range(0, len(pred[0])):
            if pred[i][j] < 0.5:
                pred[i][j] = 0
            else:
                pred[i][j] = 1
    assert(len(pred) == len(label))
    for i in range(0, len(pred)):
        assert(len(pred[i]) == 6 and len(label[i]) == 6)
        msg = f'{pred[i][0]},{pred[i][1]},{pred[i][2]},{pred[i][3]},{pred[i][4]},{pred[i][5]}, {label[i][0]},{label[i][1]},{label[i][2]},{label[i][3]},{label[i][4]},{label[i][5]}\n'
        f.write(msg)

def Eval(net, loader):
    valid_loss = 0
    valid_accuracy = 0 
    all_labels = []
    all_predictions = []
    pred_all = []
    threshold = 0.5
    with torch.no_grad():          
        # 创建一个迭代对象
        items = enumerate(loader)
        total_items = len(loader)  # 获取迭代对象的总长度  
        net.eval()   
        net = net.cuda()
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
            print_log(predictions.detach().cpu().numpy().tolist(), labels.detach().cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(pred_all)
    valid_accuracy = Accuracy(y_true, y_pred)
    overview(y_true, y_pred, y_prob, valid_loss/TEST_SIZE)
    calculate_class_metrics(y_true, y_pred)
    return valid_accuracy, valid_loss/TEST_SIZE

if __name__ == "__main__":
    leaf_model.eval() # 设置模型的状态
    test_acc, test_loss = Eval(leaf_model, loader=test_loader)
    fa.write(f'{exp_number},{model_name},{test_acc},{test_loss}\n')