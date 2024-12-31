import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import CAM, SmoothGradCAMpp
from torchcam.utils import overlay_mask
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import matplotlib.pyplot as plt
import torch
import numpy as np
from models import *

from torchvision.transforms.functional import to_pil_image

exp_number = 'swtres'
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

# 设置CAM提取器
cam_extractor = SmoothGradCAMpp(model, target_layer=model.model1.layers[3])

# 预处理输入图像
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
img_path = 'D:\Plant-Pathology-main\plant_dataset\\test\images\\f1b52245e41339fa.jpg'
img = Image.open(img_path).convert('RGB')
plt.imshow(img)
plt.axis('off')
plt.show()
input_tensor = preprocess(img).unsqueeze(0).to(device)

# 生成类激活映射图
output = model(input_tensor)
activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)

# 将类激活映射图转换为PIL图像
activation_map_img = to_pil_image(activation_map[0].cpu(), mode='F')

# 将类激活映射图叠加在输入图像上
result = overlay_mask(img, activation_map_img, alpha=0.5)

# 显示结果
plt.imshow(result)
plt.axis('off')
plt.show()