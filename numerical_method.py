import torch
import torch.nn as nn
import os
import torch.optim as optim

import numpy as np
import itertools

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


# Generate training data
x = torch.rand(1200, 25)

# Split data into training and validation sets
x_train, x_val = train_test_split(x.numpy(), test_size=0.2, random_state=42)
x_train, x_val = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_val, dtype=torch.float32)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Lists to store loss values
train_losses = []
val_losses = []

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()

    # Forward pass with training data
    inputs_train = net(x_train)
    outputs_train = non_linear(inputs_train[:, 0], inputs_train[:, 1], inputs_train[:, 2], inputs_train[:, 3], inputs_train[:, 4], inputs_train[:, 5], inputs_train[:, 6],
                             inputs_train[:, 7], inputs_train[:, 8], inputs_train[:, 9], inputs_train[:, 10], inputs_train[:, 11], inputs_train[:, 12], inputs_train[:, 13],
                             inputs_train[:, 14], inputs_train[:, 15], inputs_train[:, 16], inputs_train[:, 17], inputs_train[:, 18], inputs_train[:, 18], inputs_train[:, 20],
                             inputs_train[:, 21], inputs_train[:, 22], inputs_train[:, 23], inputs_train[:, 24])
    loss_train = criterion(outputs_train, torch.zeros(outputs_train.size(), dtype=torch.float32))
    loss_train.backward()
    optimizer.step()

    # Validation
    with torch.no_grad():
        inputs_val = net(x_val)
        outputs_val = non_linear(inputs_val[:, 0], inputs_val[:, 1], inputs_val[:, 2], inputs_val[:, 3], inputs_val[:, 4], inputs_val[:, 5], inputs_val[:, 6],
                               inputs_val[:, 7], inputs_val[:, 8], inputs_val[:, 9], inputs_val[:, 10], inputs_val[:, 11], inputs_val[:, 12], inputs_val[:, 13],
                               inputs_val[:, 14], inputs_val[:, 15], inputs_val[:, 16], inputs_val[:, 17], inputs_val[:, 18], inputs_val[:, 18], inputs_val[:, 20],
                               inputs_val[:, 21], inputs_val[:, 22], inputs_val[:, 23], inputs_val[:, 24])
        loss_val = criterion(outputs_val, torch.zeros(outputs_val.size(), dtype=torch.float32))

    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    print(f'Epoch {epoch}, Training Loss: {loss_train.item()}, Validation Loss: {loss_val.item()}')

# Plot the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()