import torch
import torchvision.models as models

# 下载出错时，需要删除下面路径对应的文件

# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" 
# to C:\Users\anders/.cache\torch\hub\checkpoints\vgg16-397923af.pth

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')