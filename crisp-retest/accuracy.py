"""Separate full-test-set quantized simulation from sampled HE/FSS validation.
This does not claim a malicious-secure two-party model-accuracy experiment.
"""
import pathlib,json,ctypes,time,sys
import numpy as np
import torch
from torch import nn
from torchvision import datasets,transforms
from train_models import ModelA,ModelB
ROOT=pathlib.Path(__file__).resolve().parent
torch.set_num_threads(4)
import subprocess,struct,atexit
worker=None
def read_exact(n):
 out=bytearray()
 while len(out)<n:
  b=worker.stdout.read(n-len(out))
  if not b:raise RuntimeError('C++ numerical worker exited')
  out.extend(b)
 return out
def request(payload):
 global worker
 if worker is None:
  worker=subprocess.Popen([str(ROOT/'build-ckks/crisp_model_worker')],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
  atexit.register(lambda: worker.terminate() if worker.poll() is None else None)
 worker.stdin.write(payload);worker.stdin.flush()
 code=struct.unpack('<i',read_exact(4))[0]
 if code:
  n=struct.unpack('<I',read_exact(4))[0];raise RuntimeError(read_exact(n).decode())
def linear(kind,h,wi,ci,co,k,pad,w,bias,x,shape):
 request(struct.pack('<7i',kind,h,wi,ci,co,k,pad)+w.tobytes()+bias.tobytes()+x.tobytes())
 err=struct.unpack('<d',read_exact(8))[0];out=np.frombuffer(read_exact(int(np.prod(shape))*8),dtype=np.float64).copy().reshape(shape)
 return out,err
def relu(x,mode):
 request(struct.pack('<4I',2,x.size,16,mode)+x.tobytes())
 return np.frombuffer(read_exact(x.size*8),dtype=np.float64).copy().reshape(x.shape)
def cryptographic_forward(model,input,mode=2):
 x=input.numpy().astype(np.float64);layers=[]
 for idx,l in enumerate(model.layers):
  start=time.monotonic();error=None
  if isinstance(l,(nn.Conv2d,nn.Linear)):
   w=np.ascontiguousarray(l.weight.detach().numpy(),dtype=np.float64)
   if isinstance(l,nn.Conv2d):
    ci,h,wi=x.shape;co=l.out_channels;k=l.kernel_size[0];pad=l.padding[0];oh=h+2*pad-k+1;ow=wi+2*pad-k+1;out=np.empty((co,oh,ow),np.float64);kind=1
   else:
    ci=l.in_features;co=l.out_features;h=wi=k=1;pad=0;out=np.empty(co,np.float64);kind=0
   x=np.ascontiguousarray(x);bias=np.ascontiguousarray(l.bias.detach().numpy(),dtype=np.float64);x,error=linear(kind,h,wi,ci,co,k,pad,w,bias,x,out.shape)
  elif isinstance(l,nn.ReLU):
   x=relu(np.ascontiguousarray(x),mode)
  elif isinstance(l,nn.MaxPool2d):
   # Paper's comparison tree: max(a,b)=b+ReLU(a-b), two rounds for 2x2.
   def maximum(a,b):
    delta=np.ascontiguousarray(a-b);out=relu(delta,mode)
    return b+out
   a=maximum(x[:,0::2,0::2],x[:,0::2,1::2]);b=maximum(x[:,1::2,0::2],x[:,1::2,1::2]);x=maximum(a,b)
  elif isinstance(l,nn.Flatten):x=x.reshape(-1)
  else:raise ValueError(type(l))
  layers.append({'layer':idx,'type':type(l).__name__,'seconds':time.monotonic()-start,'ckks_max_abs_error':error})
 return x,layers

# Full-set simulation uses actual trained weights and fixed-point rounding at
# ReLUs. It does NOT model CKKS error, FSS approximation, or network execution.
def quantized_forward(model,x):
 for l in model.layers:
  x=l(x)
  if isinstance(l,nn.ReLU):x=torch.round(x*65536)/65536
 return x
if __name__=='__main__':
 count=int(sys.argv[1]) if len(sys.argv)>1 else 16
 for label,cls,ds in [('A',ModelA,datasets.MNIST),('B',ModelB,datasets.CIFAR10)]:
  model=cls();ck=torch.load(ROOT/'models'/f'{label}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
  test=ds(str(ROOT/'data'),train=False,download=False,transform=transforms.ToTensor());loader=torch.utils.data.DataLoader(test,batch_size=128)
  print('START full-test simulation',label,flush=True)
  correct=plain=agree=0
  with torch.no_grad():
   for x,y in loader:
    p=model(x).argmax(1);q=quantized_forward(model,x).argmax(1);plain+=(p==y).sum().item();correct+=(q==y).sum().item();agree+=(p==q).sum().item()
  summary={'model':label,'source':'newly trained same-shape model','full_test_examples':len(test),'plaintext_accuracy':plain/len(test),'fixedpoint_simulation_accuracy':correct/len(test),'simulation_prediction_agreement':agree/len(test),'simulation_is_crisp_secure_inference':False,'sample_scope':'single-process CKKS linear layers + exact FSS ReLU functional validation; FSS comparison-tree maxpool and encrypted bias; not secure inference','sample_count':count,'samples':[]}
  print('START cryptographic sample',label,flush=True)
  # Deterministic spread through test set, rather than selecting successes.
  for i in np.linspace(0,len(test)-1,count,dtype=int):
   x,y=test[int(i)]
   with torch.no_grad():reference=model(x.unsqueeze(0)).squeeze().numpy()
   out,detail=cryptographic_forward(model,x,2)
   row={'index':int(i),'label':int(y),'plaintext_prediction':int(reference.argmax()),'he_fss_prediction':int(out.argmax()),'logits_max_abs_difference':float(np.max(np.abs(reference-out))),'layers':detail}
   summary['samples'].append(row);(ROOT/'results'/f'accuracy-{label}.json').write_text(json.dumps(summary,indent=2));print(label,int(i),row['plaintext_prediction'],row['he_fss_prediction'],row['logits_max_abs_difference'],flush=True)
  summary['sample_prediction_agreement']=sum(r['plaintext_prediction']==r['he_fss_prediction'] for r in summary['samples'])/count
  summary['sample_he_fss_accuracy']=sum(r['label']==r['he_fss_prediction'] for r in summary['samples'])/count
  (ROOT/'results'/f'accuracy-{label}.json').write_text(json.dumps(summary,indent=2))
