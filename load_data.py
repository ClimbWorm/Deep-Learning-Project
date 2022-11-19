#download data and unuzip
!wget https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-icon_sample.zip
!unzip -q LLD-icon_sample.zip

import glob
import torch
from PIL import Image
import torchvision.transforms as transforms

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

#read images and transform to tensor
datasets = []
images=glob.glob("5klogos/*.png")
for image in images:
    img = Image.open(image)
    tensor = transform(img)
    datasets.append(tensor)
