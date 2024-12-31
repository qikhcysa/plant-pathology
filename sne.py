import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from models import *
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from models import *
from dataset import LeafDataset
from evaltools import *

exp_number = 'swtres'
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


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载预训练模型
# model_name = "ResNet50 with DE"
# model = ResNet50(num_classes=6, pretrained=True).to(device)
# model_name = "VGG16 with DE"
# model = VGG16(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer with DE"
# model = SwinTransformer(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_STN_DE"
# model = SwinTransformer_STN(num_classes=6, pretrained=True).to(device)
model_name = "SwinTransformer_ResNet_ATT_DE"
model = SwinTransformer_ResNet_ATT(num_classes=6, pretrained=True).to(device)
# model_name = "EfficientNet_DE"
# model = EfficientNet(num_classes=6, pretrained=True).to(device)
# model_name = "MobileNetV2 with data enhancement"
# model = MobileNetV2(num_classes=6, pretrained=True).to(device)
# model_name = "Xception"
# model = Xception(num_classes=6, pretrained=True).to(device)
model_save_root_path = f'D:/Plant-Pathology-main/run/{exp_number}/best.pt'
model.load_state_dict(torch.load(model_save_root_path, weights_only=True))
model.eval()

# 参数传递
num_of_epoch = args.num_of_epoch
learning_rate = args.lr
batch_size = args.batch_size
test_label_dir = args.test_label_dir
test_image_dir = args.test_image_dir
save_dir = args.save_dir
weights_dir = args.weights_dir

test_ds = LeafDataset(csv_file=test_label_dir, imgs_path=test_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([384, 384]),
                    torchvision.transforms.Normalize(mean=[0.40568268,0.51418954,0.3238448],std=[0.2018306,0.18794753,0.18870047],),
                    ]),
                    augment=None)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=4)



# 提取特征
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        items = enumerate(dataloader)
        total_items = len(dataloader)  # 获取迭代对象的总长度
        for _, (images, targets) in tqdm(items,total=total_items,desc="val"): 
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# 应用 t-SNE 并进行可视化
def plot_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of model features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.savefig(save_path, dpi=800)
    plt.show()

# 在训练结束后进行 t-SNE 可视化
if __name__ == "__main__":
    # 假设你已经定义了验证数据加载器 val_loader
    features, labels = extract_features(model, test_loader, device)
    plot_tsne(features, labels, args.save_dir + "tsne.jpg")
