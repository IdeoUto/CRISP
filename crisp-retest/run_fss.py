import subprocess,pathlib,json,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
exe=ROOT/'build-fss/crisp_relu'
def run(n,mode='paper',rep=0,threads=4):
 dest=ROOT/'results'/f'relu-{mode}-{n}-r{rep}';dest.mkdir(exist_ok=True)
 procs=[]
 for role in ['dealer','0','1']:
  log=open(dest/f'{role}.log','w')
  p=subprocess.Popen([str(exe),role,str(n),str(threads),'23400',mode,str(dest/f'{role}.json')],stdout=log,stderr=subprocess.STDOUT)
  procs.append((p,log))
 try:
  for p,log in procs:
   code=p.wait(timeout=600);log.close()
   if code:raise RuntimeError(f'exit {code} in {dest}')
 except BaseException:
  for p,log in procs:
   if p.poll() is None:p.kill()
  raise
 data=[json.load(open(dest/f'{r}.json')) for r in ['dealer','0','1']]
 row={'n':n,'mode':mode,'repeat':rep,'offline_s':data[0]['offline_s'],'online_s':max(data[1]['online_s'],data[2]['online_s']),'offline_key_bytes':data[0]['key_distribution_bytes'],'online_bytes':sum(d['online_sent_bytes'] for d in data[1:]),'errors':data[1]['protocol_errors'],'approximation_mismatches':data[1]['signed_relu_mismatches'],'threads':threads,'network':'loopback','malicious_security_complete':False}
 with open(ROOT/'results/fss-summary.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:run(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'paper')
 else:
  for rep in range(3):
   for n in [2**k for k in range(9,19)]+[10340,173056]:run(n,'paper',rep)
