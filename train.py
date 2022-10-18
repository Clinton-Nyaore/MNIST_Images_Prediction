# Lets import the modules
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim

# Define the device to use
device = torch.device('cpu')

# We define a transform to normalize our data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# Download and load training data
path = './data'
trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
testset = datasets.MNIST(path, download=True, train=False, transform=transform)

train_data = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define the network to use
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 10))
model = model.to(device)


# Define the network, define the criterion and optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train and validate the network
epochs = 2
steps = 0  ### This I will add later  ###
train_losses, test_losses = [], []

for e in range(epochs):
  running_loss = 0

  for images, labels in train_data:
    # Flatten the images
    images = images.view(images.shape[0], -1).to(device)
    labels = labels.to(device)

    #Training pass
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  else:

    test_loss = 0
    accuracy = 0
    
    with torch.no_grad():

      for images, labels in test_data:
        # Flatten the images again
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)

        log_preds = model(images)
        test_loss += criterion(log_preds, labels)

        log_preds = torch.exp(log_preds)
        top_p, top_class = log_preds.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    train_losses.append(running_loss/len(train_data))
    test_losses.append(test_loss/len(test_data))

    print("Epoch {}/{} ... ".format(e+1, epochs),
          "Training Loss : {:.3f} ... ".format(running_loss/len(train_data)),
          "Test Loss : {:.3f} ... ".format(test_loss/len(test_data)),
          "Test Accuracy : {:.3f}".format(accuracy/len(test_data)))

print()
print('Done training our model')
print()

# Save our model 
def save_model(model):
    print('Saving our model ... ')
    save_dir = 'digits.pth'
    checkpoint = {
                'model': model.cpu(),
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    print('Done saving our model')
    return 0

save_model(model)