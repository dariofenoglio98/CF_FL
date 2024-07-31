"""
This code automatically downloads the CIFAR10 dataset and creates a custom dataset.
A ResNet18 model is used to extract features from the images, which are then saved to disk.
"""


import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights


def main():
    # Load CIFAR10 dataset
    data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=None)

    # Step 2: Prepare ResNet18 model for feature extraction
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    # Creating the custom dataset
    class CustomDSpritesDataset(torch.utils.data.Dataset):
        def __init__(self, data, transform=None):
            self.dataset = data
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx][0]
            target_label = self.dataset[idx][1]
            
            if self.transform:
                image = preprocess(image)

            return image, torch.tensor(target_label, dtype=torch.float32)

    # Create custom datasets
    custom_train_dataset = CustomDSpritesDataset(data, transform=preprocess)

    # DataLoaders
    train_loader = DataLoader(custom_train_dataset, batch_size=64, shuffle=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "mps"
    model = model.to(device)

    # Step 3: Extract features
    def extract_features(data_loader):
        features = []
        task_labels = []

        with torch.no_grad():
            for imgs, tasks in tqdm(data_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                features.append(out.cpu().numpy())
                task_labels.append(tasks.numpy())

        return np.concatenate(features), np.concatenate(task_labels)

    train_features, train_tasks = extract_features(train_loader)

    # Step 4: Save the embeddings and labels
    np.save('data/train_features_cifar10.npy', train_features)
    np.save('data/train_tasks_cifar10.npy', train_tasks)


if __name__ == "__main__":
    main()
