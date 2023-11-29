import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_gpus = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

def non_linear(b, b1, b2, b3, b4, w15, w16, w116, w11, w27, w28, w29, w22, w310, w311, w312, w33, w413, w414, w415, w44, w1, w2, w3, w4):

    # initialization
    m = torch.tensor(complex(0, torch.pi/4))
    right_side = []
    left_side = []

    # Each spin can be either -1 or +1
    options = [-1, 1]

    # Generate all possible configurations
    all_configurations = list(itertools.product(options, repeat=16))

    # Write each side of the equations based on the configurations
    for i in range(len(all_configurations)):
        conf = all_configurations[i]
        conf_tensor = torch.tensor(conf, dtype=torch.float32)

        # creating complex tensors
        z1 = torch.complex(b1, w15*conf_tensor[4] + w16*conf_tensor[5] + w116*conf_tensor[15] - w11*conf_tensor[0])
        z2 = torch.complex(b2, w27*conf_tensor[6] + w28*conf_tensor[7] + w29*conf_tensor[8] - w22*conf_tensor[1])
        z3 = torch.complex(b3, w310*conf_tensor[9] + w311*conf_tensor[10] + w312*conf_tensor[11] - w33*conf_tensor[2])
        z4 = torch.complex(b4, w413*conf_tensor[12] + w414*conf_tensor[13] + w415*conf_tensor[14] - w44*conf_tensor[3])
        z = torch.complex(b, - w1*conf_tensor[0] - w2*conf_tensor[1] - w3*conf_tensor[2] - w4*conf_tensor[3])

        a1 = torch.complex(b1, w15*conf_tensor[4] + w16*conf_tensor[5] + w116*conf_tensor[15] + w11*conf_tensor[0])
        a2 = torch.complex(b2, w27*conf_tensor[6] + w28*conf_tensor[7] + w29*conf_tensor[8] + w22*conf_tensor[1])
        a3 = torch.complex(b3, w310*conf_tensor[9] + w311*conf_tensor[10] + w312*conf_tensor[11] + w33*conf_tensor[2])
        a4 = torch.complex(b4, w413*conf_tensor[12] + w414*conf_tensor[13] + w415*conf_tensor[14] + w44*conf_tensor[3])
        a = torch.complex(b, + w1*conf_tensor[0] + w2*conf_tensor[1] + w3*conf_tensor[2] + w4*conf_tensor[3])

        # Write the right hand side of each equation
        right_side.append(torch.cosh(a)*torch.cosh(a1)*torch.cosh(a2)*torch.cosh(a3)*torch.cosh(a4)*
                         torch.cosh(m*torch.complex(conf_tensor[14] + conf_tensor[15] - conf_tensor[0] - conf_tensor[3], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[6] + conf_tensor[7] - conf_tensor[0] - conf_tensor[1], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[11] + conf_tensor[12] - conf_tensor[2] - conf_tensor[3], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[8] + conf_tensor[9] - conf_tensor[1] - conf_tensor[2], torch.tensor(0.0)))
                         )

        # write teh left hand side of each equation
        left_side.append(torch.cosh(z)*torch.cosh(z1)*torch.cosh(z2)*torch.cosh(z3)*torch.cosh(z4)*
                         torch.cosh(m*torch.complex(conf_tensor[14] + conf_tensor[15] - conf_tensor[0] - conf_tensor[3], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[6] + conf_tensor[7] - conf_tensor[0] - conf_tensor[1], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[11] + conf_tensor[12] - conf_tensor[2] - conf_tensor[3], torch.tensor(0.0)))*
                         torch.cosh(m*torch.complex(conf_tensor[8] + conf_tensor[9] - conf_tensor[1] - conf_tensor[2], torch.tensor(0.0)))
                         )

    # I think this could be improved
    left_side = torch.from_numpy(np.array(left_side)).float()
    right_side = torch.from_numpy(np.array(right_side)).float()

    # Write the optimization function
    # subs = [(a - b)**2 for a, b in zip(left_side, right_side)]
    # loss = sum(subs)

    # then we set it this way
    loss = ((left_side - right_side)**(2)).sum()
    return torch.abs(loss)

# Neural Network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 100)
        self.fc2 = nn.Linear(100, 25)

    def forward(self, x):
        x = x['input']
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the neural network
net = Net()
net = net.to(device)

# run it across several GPUs
if len(num_gpus) > 1:
    print("Let's use", len(num_gpus), "GPUs!")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)
    model = torch.nn.DataParallel(net, device_ids=num_gpus)
    model = model.module


class Aulan_Dataset(Dataset):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return {'input': torch.from_numpy(self.x[idx]).float()}, {'target': torch.from_numpy(self.y[idx]).float()}

def dict_to(x, device='cuda'): 
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

# Generate training data
n_variables = 25
n_samples = 10000
batch_size = 256
n_epochs = 1000
save_best_weights = True
best_eval = 1000

x = torch.rand(n_samples, n_variables)
y = torch.zeros((n_samples, n_variables))

# Split data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

train_dataset = Aulan_Dataset(X_train, y_train)
eval_dataset = Aulan_Dataset(X_val, y_val)

train_loader = DeviceDataLoader(DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True), device)
eval_loader = DeviceDataLoader(DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, drop_last=True), device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Lists to store loss values
train_losses = []
val_losses = []

# Training loop
for epoch in range(n_epochs):
    net.train()
    
    epoch_train_loss, epoch_val_loss = 0, 0 

    for data_train, label_train in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        inputs_train = net(data_train)        

        # Forward pass with training data
        inputs_train = net(data_train)
        outputs_train = non_linear(inputs_train[:, 0], inputs_train[:, 1], inputs_train[:, 2], inputs_train[:, 3], inputs_train[:, 4], inputs_train[:, 5], inputs_train[:, 6],
                                inputs_train[:, 7], inputs_train[:, 8], inputs_train[:, 9], inputs_train[:, 10], inputs_train[:, 11], inputs_train[:, 12], inputs_train[:, 13],
                                inputs_train[:, 14], inputs_train[:, 15], inputs_train[:, 16], inputs_train[:, 17], inputs_train[:, 18], inputs_train[:, 18], inputs_train[:, 20],
                                inputs_train[:, 21], inputs_train[:, 22], inputs_train[:, 23], inputs_train[:, 24])
        loss_train = criterion(outputs_train, label_train['target'])
        epoch_train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()

    epoch_train_loss = epoch_train_loss/ len(train_loader)

    # Validation
    net.eval()
    for data_eval, label_eval in tqdm(eval_loader, desc='Evaluating'):
        with torch.no_grad():
            inputs_val = net(data_eval)
        outputs_val = non_linear(inputs_val[:, 0], inputs_val[:, 1], inputs_val[:, 2], inputs_val[:, 3], inputs_val[:, 4], inputs_val[:, 5], inputs_val[:, 6],
                                inputs_val[:, 7], inputs_val[:, 8], inputs_val[:, 9], inputs_val[:, 10], inputs_val[:, 11], inputs_val[:, 12], inputs_val[:, 13],
                                inputs_val[:, 14], inputs_val[:, 15], inputs_val[:, 16], inputs_val[:, 17], inputs_val[:, 18], inputs_val[:, 18], inputs_val[:, 20],
                                inputs_val[:, 21], inputs_val[:, 22], inputs_val[:, 23], inputs_val[:, 24])
        loss_val = criterion(outputs_val, label_eval['target'])
        epoch_val_loss += loss_val.item()

        epoch_val_loss = epoch_val_loss / len(eval_loader)
    

    if save_best_weights:
        if epoch_val_loss < best_eval:
            best_eval = epoch_val_loss
            torch.save({'checkpoint': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'best_model.pth')
            print('Epoch {}/{} - Training Loss: {} - New Validation Loss: {}'.format(epoch, n_epochs, epoch_train_loss, epoch_val_loss))
        else:
            print(f'Epoch {epoch}/{n_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}')

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

# Plot the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
