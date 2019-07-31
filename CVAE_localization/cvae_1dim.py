import torch

import torch
import torch.utils.data
import torch.distributions as dist
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

sample_c1=torch.ones(2000)*-1.0
sample_c2=torch.ones(2000)*3.0
sample_c3=torch.ones(2000)*5.0
sample_c=torch.cat([sample_c1,sample_c2,sample_c3],dim=0)
sample_s=dist.Normal(sample_c,0.5).sample()
sample_o=dist.Normal(sample_s,0.5).sample()

train_c=sample_c
train_s=sample_s
train_o=sample_o

test_c=sample_c
test_s=sample_s
test_o=sample_o

kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}

train = torch.utils.data.TensorDataset(train_o, train_c)
train_loader = torch.utils.data.DataLoader(train, shuffle=False, **kwargs)
test = torch.utils.data.TensorDataset(test_o, test_c)
test_loader = torch.utils.data.DataLoader(test, shuffle=False, **kwargs)

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

    plot_number = 1
    writer = SummaryWriter()
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        writer.add_scalar('train_loss', train_loss.item(), epoch)
        writer.add_scalar('test_loss', test_loss.item(), epoch)
    writer.close()
