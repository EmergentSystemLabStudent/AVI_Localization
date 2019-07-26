import torch

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm


from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE

device = "cpu"
batch_size = 128
epochs =50
seed = 1
torch.manual_seed(seed)

o_dim = 1
c_dim = 1
s_dim = 1

class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(cond_var=["o","c"], var=["s"], name="q")
        self.fc1 = nn.Linear(o_dim+c_dim, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc31 = nn.Linear(4, s_dim)
        self.fc32 = nn.Linear(4, s_dim)
    def forward(self, o, c):
        h = F.relu(self.fc1( torch.stack([o,c],1) ))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

class Generator(Normal):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["s"], var=["o"], name="p")
        self.fc1 = nn.Linear(s_dim, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, o_dim)
    def forward(self, s):
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc3(h),"scale":torch.tensor(0.3).to(device)}

class prior_set(Normal):
    def forward(self, c):
        return{"loc":c, "scale":torch.tensor(0.3).to(device)}

def train(epoch):
    train_loss = 0
    for o,c in tqdm(train_loader):
        o = o.to(device)
        c = c.to(device)
        loss = model.train({"o": o, "c": c})
        train_loss += loss
    train_loss = train_loss * train_loader.batch_size / len(dc)
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def test(epoch):
    test_loss = 0
    for o, c in test_loader:
         o = o.to(device)
         c = c.to(device)
         loss = model.test({"o": o, "c": c})
         test_loss += loss
    test_loss = test_loss * test_loader.batch_size / len(tc)
    print('Test loss: {:.4f}'.format(test_loss))
    return test_loss

    
if __name__ == '__main__':
    p = Generator()
    q = Inference()
    p.to(device)
    q.to(device)
    print(p)
    print(q)
    prior = prior_set()
    prior.to(device)
    print(prior)
    kl = KullbackLeibler(q, prior)
    print(kl)
    model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr":1e-3})
    print(model)
    
