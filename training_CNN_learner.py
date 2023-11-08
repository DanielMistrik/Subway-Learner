import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm

import CNNLearner
import subway_dataset

criterion = nn.CrossEntropyLoss()


def train_learner_given_dataloader(data_loader, num_epochs=20):
    model = CNNLearner.SubwayCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(data_loader, leave=False):
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
        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model


def get_trained_learner():
    train_loader = DataLoader(subway_dataset.SubwayDataset(), batch_size=32, shuffle=True)
    return train_learner_given_dataloader(train_loader)

if __name__ == '__main__':
    model = get_trained_learner()
    torch.save(model, 'sl-model.pth')