import subprocess,pathlib,re,json,sys
ROOT=pathlib.Path(__file__).resolve().parent

def run(n,rep=0):
 dest=ROOT/'results'/f'simc-ring32-{n}-r{rep}';dest.mkdir(exist_ok=True);ps=[]
 for p in [1,2]:
  f=open(dest/f'{p}.log','w');ps.append((subprocess.Popen([str(ROOT/'build-ckks/simc_ring32'),str(p),'127.0.0.1','25100','32','1',str(n),'4'],stdout=f,stderr=subprocess.STDOUT),f))
 try:
  for p,f in ps:
   ret=p.wait(timeout=900);f.close()
   if ret:raise RuntimeError(f'{dest}: exit {ret}')
 except BaseException:
  for p,f in ps:
   if p.poll() is None:p.kill()
  raise
 logs=[(dest/f'{p}.log').read_text() for p in [1,2]]
 def grab(pat):
  vals=[]
  for t in logs:
   m=re.search(pat,t)
   if not m:return None
   vals.append(int(m[1]))
  return sum(vals)
 setup_bytes=grab(r'Setup Data \(Bytes\): (\d+)')
 circuit_bytes=grab(r'Circuit Data \(Bytes\): (\d+)')
 row={'n':n,'repeat':rep,'total_gc_s':max(int(re.search(r'Time Taken: (\d+)',t)[1]) for t in logs)/1e6,'total_bytes':sum(int(re.search(r'Sent Data \(Bytes\): (\d+)',t)[1]) for t in logs),'setup_bytes':setup_bytes,'circuit_bytes':circuit_bytes,'scope':'SIMC GC table technique adapted to ring32, includes GC/OT setup and EMP circuit table generation; not offline-separated original SIMC','correct_outputs':int(re.search(r'Correct Outputs: (\d+)',logs[0])[1])}
 with open(ROOT/'results/baseline-summary.jsonl','a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:run(int(sys.argv[1]))
 else:
  for rep in range(3):
   for k in range(9,19):run(2**k,rep)
