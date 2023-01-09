import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
from config.data_config import INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import cv2


def default_loss():
    return nn.CrossEntropyLoss()


def default_optimizer(model, learning_rate = 0.001):
    return optim.Adam(model.parameters(), lr = learning_rate)


def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def model_prep_and_summary(model, device):
    """
    Move model to GPU and print model summary
    """
    # Define the model and move it to GPU:
    model = model
    model = model.to(device)
    print('Current device: ' + str(device))
    print('Is Model on CUDA: ' + str(next(model.parameters()).is_cuda))
    # Display model summary:
    summary(model, (INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))


class MyCnnModel(nn.Module):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Define a max pooling layer to use repeatedly in the forward function
        # The role of pooling layer is to reduce the spatial dimension (H, W) of the input volume for next layers.
        # It only affects weight and height but not depth.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # output shape of maxpool3 is 64*28*28
        self.fc14 = nn.Linear(64*28*28, 500)
        self.fc15 = nn.Linear(500, 50)
        # output of the final DC layer = 6 = number of classes
        self.fc16 = nn.Linear(50, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # maxpool1 output shape is 16*112*112 (112 = (224-2)/2 + 1)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # maxpool2 output shape is 32*56*56 (56 = (112-2)/2 + 1)
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        # maxpool3 output shape is 64*28*28 (28 = (56-2)/2 + 1)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        # x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    # device = get_default_device()
    model = model.to(device)
    train_result_dict = {'epoch': [], 'train_loss': [],
                         'val_loss': [], 'accuracy': [], 'time': []}

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        model.train()  # set the model to training mode, parameters are updated
        for i, data in enumerate(train_loader, 0):
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(image)  # forward propagation
            loss = criterion(outputs, class_index)  # loss calculation
            loss.backward()  # backward propagation
            optimizer.step()  # params update
            train_loss += loss.item()  # loss for each minibatch
            _, predicted = torch.max(outputs.data, 1)
            total += class_index.size(0)
            correct += (predicted == class_index).sum().item()
            epoch_accuracy = round(float(correct)/float(total)*100, 2)

        # Here evaluation is combined together with
        val_loss = 0.0
        model.eval()  # set the model to evaluation mode, parameters are frozen
        for i, data in enumerate(val_loader, 0):
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            loss = criterion(outputs, class_index)
            val_loss += loss.item()

        # print statistics every 1 epoch
        # divide by the length of the minibatch because loss.item() returns the loss of the whole minibatch
        train_loss_result = round(train_loss / len(train_loader), 3)
        val_loss_result = round(val_loss / len(val_loader), 3)

        epoch_time = round(time.time() - start_time, 1)
        # add statistics to the dictionary:
        train_result_dict['epoch'].append(epoch + 1)
        train_result_dict['train_loss'].append(train_loss_result)
        train_result_dict['val_loss'].append(val_loss_result)
        train_result_dict['accuracy'].append(epoch_accuracy)
        train_result_dict['time'].append(epoch_time)

        print(f'Epoch {epoch+1} \t Training Loss: {train_loss_result} \t Validation Loss: {val_loss_result} \t Epoch Train Accuracy (%): {epoch_accuracy} \t Epoch Time (s): {epoch_time}')
    # return the trained model and the loss dictionary
    return model, train_result_dict


def visualize_training(train_result_dictionary):
    # Define Data
    df = pd.DataFrame(train_result_dictionary)
    x = df['epoch']
    data_1 = df['train_loss']
    data_2 = df['val_loss']
    data_3 = df['accuracy']

    # Create Plot
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(x, data_1, color='red', label='training loss')
    ax1.plot(x, data_2, color='blue', label='validation loss')

    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.plot(x, data_3, color='green', label='Training Accuracy')

    # Add label
    plt.ylabel('Accuracy')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center')

    # Show plot
    plt.show()


def infer(model, device, data_loader):
    '''
    Calculate predicted class indices of the data_loader by the trained model 
    '''
    model = model.to(device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in data_loader:
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)
            class_index = class_index.data.cpu().numpy()
            y_true.extend(class_index)
    return y_pred, y_true


def infer_single_image(model, device, image_path, transform):
    '''
    Calculate predicted class index of the image by the trained model 
    '''
    # Prepare the Image
    image = cv2.imread(image_path)  # read image by cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform(image)
    plt.imshow(image_transformed.permute(1, 2, 0))
    image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)

    # Inference
    model.eval()
    with torch.no_grad():
        image_transformed_sq = image_transformed_sq.to(device)
        output = model(image_transformed_sq)
        _, predicted_class_index = torch.max(output.data, 1)
    print(f'Predicted Class Index: {predicted_class_index}')
    return predicted_class_index
