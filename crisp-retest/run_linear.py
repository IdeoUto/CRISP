import pathlib,subprocess,json,sys
ROOT=pathlib.Path(__file__).resolve().parent
EXE=ROOT/'build-ckks/crisp_linear'
def run(kind,model,index,degree=8192,rep=0):
 dest=ROOT/'results'/f'{kind}-{model}-{index}-N{degree}-r{rep}';dest.mkdir(exist_ok=True);procs=[]
 for role in ['0','1']:
  log=open(dest/f'{role}.log','w');p=subprocess.Popen([str(EXE),role,kind,model,str(degree),'4','24100',str(dest/f'{role}.json'),str(index)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
 try:
  for p,f in procs:
   code=p.wait(timeout=1800);f.close()
   if code:raise RuntimeError(f'{dest}: process exited {code}')
 except BaseException:
  for p,f in procs:
   if p.poll() is None:p.kill()
  raise
 data=[json.load(open(dest/f'{r}.json')) for r in ['0','1']]
 row={'kind':kind,'model':model,'layer':index,'degree':degree,'repeat':rep,'setup_s':max(d['setup_s'] for d in data),'plan_preparation_s':max(d['plan_preparation_s'] for d in data),'mask_preparation_s':max(d['mask_preparation_s'] for d in data),'initial_layer':data[0]['initial_layer'],'serialization':data[0]['serialization'],'rotation_key_policy':data[0]['rotation_key_policy'],'online_s':max(d['online_s'] for d in data),'setup_bytes':sum(d['setup_sent_bytes'] for d in data),'online_bytes':sum(d['online_sent_bytes'] for d in data),'max_abs_error':data[1]['max_abs_error'],'tag_max_abs_error':data[1]['tag_max_abs_error'],'scope':'component arithmetic benchmark; not complete malicious-secure inference'}
 with open(ROOT/'results/linear-summary.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:run(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 8192)
 else:
  for rep in range(3):
   for degree in [8192,16384]:
    for model,n in [('A',2),('B',1)]:
     for i in range(n):run('fc',model,i,degree,rep)
   for model,n in [('A',2),('B',7)]:
    for i in range(n):run('conv',model,i,8192,rep)
