import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain1, inputPathTrain2, targetPathTrain, input_size=128, target_size=256):
        super(MyTrainDataSet, self).__init__()

        self.inputPath1 = inputPathTrain1
        self.inputImages1 = os.listdir(inputPathTrain1)

        self.inputPath2 = inputPathTrain2
        self.inputImages2 = os.listdir(inputPathTrain2)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.input_size = input_size
        self.target_size = target_size

        assert len(self.inputImages1) == len(self.inputImages2) == len(self.targetImages), "Datasets must have the same length"

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):
        inputImagePath1 = os.path.join(self.inputPath1, self.inputImages1[index])
        inputImage1 = np.load(inputImagePath1).astype(np.float32)

        inputImagePath2 = os.path.join(self.inputPath2, self.inputImages2[index])
        inputImage2 = np.load(inputImagePath2).astype(np.float32)

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = np.load(targetImagePath).astype(np.float32)

        inputImage1 = torch.from_numpy(inputImage1).unsqueeze(0)  # Add channel dimension
        inputImage2 = torch.from_numpy(inputImage2).unsqueeze(0)  # Add channel dimension
        targetImage = torch.from_numpy(targetImage).unsqueeze(0)  # Add channel dimension

        inputImage = torch.cat((inputImage1, inputImage2), dim=0)  # Concatenate along the channel dimension

        return inputImage, targetImage


class MyValDataSet(Dataset):
    def __init__(self, inputPathTrain1, inputPathTrain2, targetPathTrain, input_size=128, target_size=256):
        super(MyValDataSet, self).__init__()

        self.inputPath1 = inputPathTrain1
        self.inputImages1 = os.listdir(inputPathTrain1)

        self.inputPath2 = inputPathTrain2
        self.inputImages2 = os.listdir(inputPathTrain2)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.input_size = input_size
        self.target_size = target_size

        assert len(self.inputImages1) == len(self.inputImages2) == len(self.targetImages), "Datasets must have the same length"

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):
        inputImagePath1 = os.path.join(self.inputPath1, self.inputImages1[index])
        inputImage1 = np.load(inputImagePath1).astype(np.float32)

        inputImagePath2 = os.path.join(self.inputPath2, self.inputImages2[index])
        inputImage2 = np.load(inputImagePath2).astype(np.float32)

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = np.load(targetImagePath).astype(np.float32)

        inputImage1 = torch.from_numpy(inputImage1).unsqueeze(0)  # Add channel dimension
        inputImage2 = torch.from_numpy(inputImage2).unsqueeze(0)  # Add channel dimension
        targetImage = torch.from_numpy(targetImage).unsqueeze(0)  # Add channel dimension

        inputImage = torch.cat((inputImage1, inputImage2), dim=0)  # Concatenate along the channel dimension

        return inputImage, targetImage


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTrain1, inputPathTrain2, targetPathTrain, input_size=128, target_size=256):
        super(MyTestDataSet, self).__init__()

        self.inputPath1 = inputPathTrain1
        self.inputImages1 = os.listdir(inputPathTrain1)

        self.inputPath2 = inputPathTrain2
        self.inputImages2 = os.listdir(inputPathTrain2)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.input_size = input_size
        self.target_size = target_size

        assert len(self.inputImages1) == len(self.inputImages2) == len(self.targetImages), "Datasets must have the same length"

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):
        inputImagePath1 = os.path.join(self.inputPath1, self.inputImages1[index])
        inputImage1 = np.load(inputImagePath1).astype(np.float32)

        inputImagePath2 = os.path.join(self.inputPath2, self.inputImages2[index])
        inputImage2 = np.load(inputImagePath2).astype(np.float32)

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = np.load(targetImagePath).astype(np.float32)

        inputImage1 = torch.from_numpy(inputImage1).unsqueeze(0)  # Add channel dimension
        inputImage2 = torch.from_numpy(inputImage2).unsqueeze(0)  # Add channel dimension
        targetImage = torch.from_numpy(targetImage).unsqueeze(0)  # Add channel dimension

        inputImage = torch.cat((inputImage1, inputImage2), dim=0)  # Concatenate along the channel dimension

        return inputImage, targetImage










