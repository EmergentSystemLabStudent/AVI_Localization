from make_data import make_data

input,output=make_data()

for i in range(99):
    print(i,input[i],output[i])



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

from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE

x_dim=2
y_dim=1
z_dim=2

class Inference(Normal):
    def __init__(self):
         super(Inference,self).__init__(cond_var=["x","y"],var=["z"],name="q")
         self.fc1 = nn.Linear(3,3)
         self.fc21 = nn.Linear(3,2)
         self.fc22 = nn.Linear(3,2)
    def forward(self,x,y):
        h = F.relu(self.fc1(tourch.cat([x,y], 1)))
        return {"loc":self.fc21(h),"scale":F.softplus(self.fc22(h))}

inference = Inference()

class Generator(Normal):
    def __init__(self):
        super(Generator, self).__init__(cond_var=["z","y"], var=["x"], name="p")
        self.fc1 = nn.Linear(3,3)
        self.fc21 = nn.Linear(3,2)
        self.fc22 = nn.Linear(3,2)
    def forward(self,z,y):
        h = F.relu(self.fc1(tourch.cat([z,y], 1)))
        return {"loc":self.fc21(h),"scale":F.softplus(self.fc22(h))}

generator = Generator()

