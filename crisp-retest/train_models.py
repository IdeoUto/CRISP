import os,json,time,pathlib,hashlib,random
import numpy as np
import torch
from torch import nn
from torchvision import datasets,transforms
ROOT=pathlib.Path(__file__).resolve().parent
random.seed(20261121);np.random.seed(20261121);torch.manual_seed(20261121);torch.set_num_threads(4)
class ModelA(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(1,16,5),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,16,5),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(256,100),nn.ReLU(),nn.Linear(100,10))
 def forward(self,x):return self.layers(x)
class ModelB(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,1),nn.ReLU(),nn.Conv2d(64,16,1),nn.ReLU(),nn.Flatten(),nn.Linear(1024,10))
 def forward(self,x):return self.layers(x)
def evaluate(model,loader,device):
 model.eval();correct=total=0
 with torch.no_grad():
  for x,y in loader:
   pred=model(x.to(device)).argmax(1).cpu();correct+=(pred==y).sum().item();total+=len(y)
 return correct/total
if __name__=='__main__':
 (ROOT/'models').mkdir(exist_ok=True);device='mps' if torch.backends.mps.is_available() else 'cpu';print('device',device,flush=True)
 for label,cls,ds,epochs in [('A',ModelA,datasets.MNIST,10),('B',ModelB,datasets.CIFAR10,30)]:
  transform=transforms.ToTensor()
  train=ds(str(ROOT/'data'),train=True,download=True,transform=transform);test=ds(str(ROOT/'data'),train=False,download=True,transform=transform)
  train,val=torch.utils.data.random_split(train,[len(train)-5000,5000],generator=torch.Generator().manual_seed(20261121));valloader=torch.utils.data.DataLoader(val,batch_size=256,num_workers=0)
  loader=torch.utils.data.DataLoader(train,batch_size=128,shuffle=True,num_workers=0);testloader=torch.utils.data.DataLoader(test,batch_size=256,num_workers=0)
  model=cls().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[15,24],gamma=.3);best=0;records=[]
  for epoch in range(epochs):
   start=time.monotonic();model.train()
   for x,y in loader:
    opt.zero_grad();loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));loss.backward();opt.step()
   scheduler.step();acc=evaluate(model,valloader,device);row={'model':label,'epoch':epoch+1,'validation_accuracy':acc,'seconds':time.monotonic()-start,'device':device};records.append(row);print(json.dumps(row),flush=True)
   if acc>best:
    best=acc;torch.save({'state_dict':model.cpu().state_dict(),'architecture':label,'epoch':epoch+1,'validation_accuracy':acc,'seed':20261121,'source':'locally trained; not original paper weights'},ROOT/'models'/f'{label}.pt');model.to(device)
   (ROOT/'models'/f'{label}-training.json').write_text(json.dumps(records,indent=2))
  # Export held-out inputs and all weights for independent C++/CKKS checking.
  ck=torch.load(ROOT/'models'/f'{label}.pt',map_location='cpu',weights_only=True);model=cls();model.load_state_dict(ck['state_dict']);model.eval()
  accuracy=evaluate(model,testloader,'cpu');(ROOT/'models'/f'{label}-test.json').write_text(json.dumps({'test_accuracy':accuracy,'test_examples':len(test),'checkpoint_selection':'validation split, not test set','source':'locally trained same-shape architecture'},indent=2))
  np.savez(ROOT/'models'/f'{label}-weights.npz',**{k:v.numpy() for k,v in model.state_dict().items()})
  x=torch.stack([test[i][0] for i in range(128)]);y=np.array([test[i][1] for i in range(128)],dtype=np.int64);np.savez(ROOT/'models'/f'{label}-test128.npz',inputs=x.numpy(),labels=y,logits=model(x).detach().numpy())
