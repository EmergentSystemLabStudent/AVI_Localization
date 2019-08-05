from make_data import make_data

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from tqdm import tqdm

batch_size = 1
epochs = 10
seed = 42
torch.manual_seed(seed)
device = "cpu"

datas=make_data()
datas=torch.Tensor(datas)

train = torch.utils.data.TensorDataset(datas)
train_loader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)


from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE


class Inference(Normal):
    def __init__(self):
         super(Inference,self).__init__(cond_var=["x","z_prev"],var=["z"],name="q")
         self.fc1 = nn.Linear(3,3)
         self.fc21 = nn.Linear(3,2)
         self.fc22 = nn.Linear(3,2)
    def forward(self,x,z_prev):
        h = F.relu(self.fc1(torch.cat([x,z_prev])))
        return {"loc":self.fc21(h),"scale":F.softplus(self.fc22(h))}

class Generator(Normal):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z"], var=["x"], name="p")
        self.fc1 = nn.Linear(2,3)
        self.fc21 = nn.Linear(3,2)
        self.fc22 = nn.Linear(3,2)
    def forward(self,z):
        h = F.relu(self.fc1(z))
        return {"loc":self.fc21(h),"scale":F.softplus(self.fc22(h))}

class Prior(Normal):
    def forward(self, z_prev):
        return{"loc": z_prev+torch.tensor([1.0,1.0]),"scale":torch.tensor([0.1,0.1])}

z_dim = 2
prior = Prior(cond_var=["z_prev"],var=["z"])

inference = Inference()
generator = Generator()

kl = KullbackLeibler(inference,prior)
model = VAE(inference,generator,regularizer=kl,optimizer=optim.Adam,optimizer_params={"lr":1e-3})

epochs=1000

def train(epoch):
    train_loss = 0
    for x in tqdm(train_loader):
        z_prev=x[0][0][0:2].to(device)
        x=x[0][0][2:3].to(device)
        #print(x,z_prev)
        loss = model.train({"z_prev":z_prev,"x":x})
        train_loss+=loss
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))    
    return train_loss

def test(epoch):
    test_loss = 0
    for x in tqdm(train_loader):
        z_prev=x[0][0][0:2].to(device)
        x=x[0][0][2:3].to(device)
        #print(x,z_prev)
        loss = model.test({"z_prev":z_prev,"x":x})
        test_loss+=loss
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))    
    return test_loss

def localize():
    for x in tqdm(train_loader):
        z_prev=x[0][0][0:2].to(device)
        x=x[0][0][2:3].to(device)
        z=inference.sample({"z_prev":z_prev,"x":x},return_all=False)
        print(z_prev,x,z)


for epoch in range(1,epochs+1):
    train_loss=train(epoch)
    test_loss=test(epoch)

localize()


