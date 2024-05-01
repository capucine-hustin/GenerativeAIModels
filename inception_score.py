import os
import numpy as np
from scipy.stats import entropy
from PIL import Image

import torch
from torch import nn
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms

from inception_model_train import modify_inception_v3


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load the pretrained inception model
    inception_model = modify_inception_v3()
    inception_model.load_state_dict(torch.load('mnist_inception_v3.pth'))
    #inception_model = inception_v3(weights=True, transform_input=False).type(dtype)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)    
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 10))          # Modified inception model only classifies 10 jclasses
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        #print("Batch tensor shape:", batch.shape)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),                # Resize images to fit Inception model input
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for Inception
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('L')  # Ensure image is loaded in grayscale
        image = self.transform(image)
        return image
    

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index]

    def __len__(self):
        return len(self.orig)   


if __name__ == '__main__':
    
    # Your folder path
    image_folder_path = 'data\generated_images_basic_unet\generated_images_basic_unet'

    # load your images here
    mnist_gen = CustomMNISTDataset(img_dir = image_folder_path)

    dataset_wrapper = IgnoreLabelDataset(mnist_gen)

    print ("Calculating Inception Score...")
    print (inception_score(dataset_wrapper, cuda=True, batch_size=32, resize=True, splits=10))
