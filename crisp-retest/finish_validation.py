import pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
p=ROOT/'results/linear-summary.jsonl'
if p.exists():p.rename(ROOT/'results'/f'linear-before-initlin-compression-{int(time.time())}.jsonl')
subprocess.run([sys.executable,str(ROOT/'run_linear.py')],check=True)
for n in [512,10340,173056]:
 subprocess.run([sys.executable,str(ROOT/'run_fss.py'),str(n),'exact'],check=True)
subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'accuracy.py'),'16'],check=True)
