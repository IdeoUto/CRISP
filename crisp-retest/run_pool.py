import subprocess,pathlib,json
ROOT=pathlib.Path(__file__).resolve().parent
for rep in range(3):
 dest=ROOT/'results'/f'pool-A-r{rep}';dest.mkdir(exist_ok=True);ps=[]
 for role in ['dealer','0','1']:
  f=open(dest/f'{role}.log','w');p=subprocess.Popen([str(ROOT/'build-fss/crisp_pool'),role,'2560','4','26100',str(dest/f'{role}.json')],stdout=f,stderr=subprocess.STDOUT);ps.append((p,f))
 try:
  for p,f in ps:
   code=p.wait(timeout=300);f.close()
   if code:raise RuntimeError(f'pool exit {code}')
 except BaseException:
  for p,f in ps:
   if p.poll() is None:p.kill()
  raise
 d=[json.load(open(dest/f'{role}.json')) for role in ['dealer','0','1']]
 row={'repeat':rep,'groups':2560,'inputs':10240,'comparisons':7680,'offline_s':d[0]['offline_s'],'offline_key_bytes':d[0]['key_bytes'],'online_s':max(d[1]['online_s'],d[2]['online_s']),'online_bytes':d[1]['online_sent_bytes']+d[2]['online_sent_bytes'],'errors':d[1]['validation_errors'],'network':'loopback','threads':4,'scope':'two pooling layers batched for component benchmark, paper approximation; not complete secure network'}
 with open(ROOT/'results/pool-summary.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
