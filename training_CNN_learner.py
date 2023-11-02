import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm

import CNNLearner
import subway_dataset

# Initializing CNN Learner and Data Loader
dataset = subway_dataset.SubwayDataset()
batch_size = 32  # You can change this to whatever batch size you want
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 20

criterion = nn.CrossEntropyLoss()


def get_trained_learner():
    model = CNNLearner.SubwayCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            one_hot_encoded_labels = np.zeros((labels.shape[0], 5))
            one_hot_encoded_labels[np.arange(labels.shape[0]), labels.to(torch.int32)] = 1
            one_hot_encoded_labels = one_hot_encoded_labels.astype(np.float32)
            labels = torch.from_numpy(one_hot_encoded_labels)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model

if __name__ == '__main__':
    model = get_trained_learner()
    torch.save(model, 'sl-model.pth')