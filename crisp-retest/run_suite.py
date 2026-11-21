import pathlib,subprocess,sys,json,platform,time
ROOT=pathlib.Path(__file__).resolve().parent
for name in ['fss','linear','baseline','pool']:
 p=ROOT/'results'/f'{name}-summary.jsonl'
 if p.exists():p.rename(ROOT/'results'/f'{name}-smoke-{int(time.time())}.jsonl')
meta={'started_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'platform':platform.platform(),'machine':platform.machine(),'network':'127.0.0.1 TCP loopback, TCP_NODELAY; no bandwidth shaping','threads':4,'repeats':3,'training_concurrent':False,'scope':'components only; malicious-secure end-to-end incomplete'}
for key,cmd in [('cpu',['sysctl','-n','machdep.cpu.brand_string']),('memory',['sysctl','-n','hw.memsize'])]:
 meta[key]=subprocess.check_output(cmd,text=True).strip()
(ROOT/'results/environment.json').write_text(json.dumps(meta,indent=2))
for script in ['run_fss.py','run_linear.py','run_baseline.py','run_pool.py']:
 print('START',script,flush=True)
 subprocess.run([sys.executable,str(ROOT/script)],check=True)
 print('FINISH',script,flush=True)
for n in [512,10340,173056]:
 subprocess.run([sys.executable,str(ROOT/'run_fss.py'),str(n),'exact'],check=True)
