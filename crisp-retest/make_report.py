import json,pathlib,statistics as st,collections,csv,hashlib,sys
ROOT=pathlib.Path(__file__).resolve().parent;R=ROOT/'results'
def read(name):return [json.loads(l) for l in (R/name).read_text().splitlines() if l.strip()]
def groups(rows,keys):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r[k] for k in keys)].append(r)
 return d
def med(rows,k):return st.median(r[k]for r in rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def writecsv(name,headers,rows):
 with (R/name).open('w',newline='')as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)
fss=read('fss-summary.jsonl');linear=read('linear-summary.jsonl');base=read('baseline-summary.jsonl');pool=read('pool-summary.jsonl')
f=groups([x for x in fss if x['mode']=='paper'],['n']);b=groups(base,['n']);l=groups(linear,['kind','model','degree','layer'])
assert all(len(v)==3 for v in f.values()) and all(len(v)==3 for v in l.values()) and all(len(v)==3 for v in b.values())
assert len(pool)==3
assert all(x['errors']==0 for x in fss+pool)
assert all(x['correct_outputs']==x['n']for x in base)
assert all(x['max_abs_error']<.005 and x['tag_max_abs_error']<.005 for x in linear)
rows=[]
for n in [2**k for k in range(9,19)]:
 v=f[(n,)];baseline=b[(n,)]
 rows.append([n,med(v,'offline_s')*1000,med(v,'online_s')*1000,med(v,'offline_key_bytes'),med(v,'online_bytes'),med(baseline,'total_gc_s')*1000,med(baseline,'total_bytes'),sum(x['approximation_mismatches']for x in v)])
writecsv('relu_metrics.csv',['relu_count','fss_offline_ms_median','fss_online_ms_median','fss_offline_key_bytes_both_parties','fss_online_bytes_both_parties','adapted_simc_gc_total_ms_median','adapted_simc_gc_total_bytes','paper_approx_mismatches_across_repeats'],rows)
linear_rows=[]
for key,v in sorted(l.items()):linear_rows.append([*key,med(v,'setup_s'),med(v,'plan_preparation_s'),med(v,'mask_preparation_s'),med(v,'online_s'),med(v,'setup_bytes'),med(v,'online_bytes'),max(x['max_abs_error']for x in v)])
writecsv('linear_metrics.csv',['kind','model','degree','layer','setup_s_median','plan_prep_s_median','mask_prep_s_median','online_s_median','setup_bytes_median','online_bytes_median','max_abs_error'],linear_rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ns=[r[0]for r in rows]
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(8,4.5))
ax.plot(ns,[r[2]for r in rows],'o-',label='FSS online')
ax.plot(ns,[r[1]+r[2]for r in rows],'s-',label='FSS offline + online')
ax.plot(ns,[r[5]for r in rows],'^-',label='Adapted SIMC GC (includes setup)')
ax.set(xscale='log',yscale='log',xlabel='Number of ReLUs',ylabel='Median milliseconds (3 repeats)',title='Component measurements, 4 threads, loopback')
ax.grid(True,which='both',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/'relu_time.png');fig.savefig(R/'relu_time.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(ns,[r[4]/2**20 for r in rows],'o-',label='FSS online, both directions');ax.plot(ns,[(r[3]+r[4])/2**20 for r in rows],'s-',label='FSS keys + online');ax.plot(ns,[r[6]/2**20 for r in rows],'^-',label='Adapted SIMC GC, including setup')
ax.set(xscale='log',yscale='log',xlabel='Number of ReLUs',ylabel='MiB',title='Actual serialized payload, both online parties');ax.grid(True,which='both',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/'relu_communication.png');fig.savefig(R/'relu_communication.svg');plt.close(fig)
if '--components-only' in sys.argv:
 print('Wrote component CSVs and figures; model report not generated')
 sys.exit(0)
accuracy={m:json.loads((R/f'accuracy-{m}.json').read_text())for m in ['A','B']}
assert all(a['sample_count']==len(a['samples'])==16 for a in accuracy.values())
env=json.loads((R/'environment.json').read_text())
summary={'environment':env,'fss_key_bytes_per_party_paper':1744,'fss_key_bytes_per_party_exact':3456,'fss_runs':len(fss),'linear_runs':len(linear),'baseline_runs':len(base),'pool_runs':len(pool),'accuracy':{m:{k:v for k,v in a.items()if k!='samples'}for m,a in accuracy.items()}}
(R/'summary.json').write_text(json.dumps(summary,indent=2))
report='''# CRISP 组件实验重测结果

## 环境

- 平台：{platform}；{cpu}；内存 {memory_gib} GiB；SEAL 3.6.4，OpenSSL 3.6.0
- 两个独立在线进程 + 独立离线 dealer，{network}
- {threads} 个工作线程，性能测试每组 {repeats} 次取中位数；模型训练不与性能测量重叠
- CKKS：N=8192/16384，模数链 [60,40,40,60]，scale=2^40，SEAL tc128，槽数 4096/8192
- 字节按 MiB=2^20 计；在线通信为双方发送量之和

## 输入维度

A：Conv(28×28,1→16,5×5,valid)，Pool2×2，Conv(12×12,16→16,5×5,valid)，Pool2×2，FC256→100→10；ReLU 总数 10340。

B：Conv(32×32,3→64,3×3,same)×2，Pool2×2，Conv(16×16,64→64,3×3,same)×2，Pool2×2，Conv(8×8,64→64,3×3,same)，Conv(8×8,64→64,1×1)，Conv(8×8,64→16,1×1)，FC1024→10；ReLU 总数 173056。

## ReLU

{relu_table}

![计算时间](results/relu_time.png)

![实际通信](results/relu_communication.png)

每方每 ReLU 预处理密钥 1744 B（精确模式 3456 B），在线两方合计 8 B；精确模式在线通信不变。单个比较密钥 856 B = 16 B 种子 + 32×(16 B 种子校正 + 8 B 输出校正 + 2 B 控制位) + 8 B 最终校正。

{prep_table}

## 全连接

下表按每个 benchmark 包含的 FC 层求和（组件和，不含 setup）。

{fc_table}

## 网络组件

{net_table}

## MaxPool（A 网络两层池化：10240 输入 / 2560 窗 / 7680 比较）

{pool_table}

## 模型准确率

{acc_table}

## 复现

```sh
./build.sh
python3 run_suite.py
.venv/bin/python accuracy.py 16
.venv/bin/python make_report.py
```

数据/权重可由 `.venv/bin/python train_models.py` 重新生成；数据下载地址、校验和及模型来源在 DATA_MODEL_PROVENANCE.json，机器可读实验配置在 EXPERIMENTS.json。
'''
report=report.format(platform=env['platform'],cpu=env['cpu'],memory_gib=round(int(env['memory'])/2**30,1),network=env['network'],threads=env['threads'],repeats=env['repeats'],relu_table=table(['ReLU数','FSS离线ms','FSS在线ms','FSS离线keys MiB','FSS在线KiB','GC总ms','GC总MiB','近似错数(3次)'],[[r[0],f'{r[1]:.3f}',f'{r[2]:.3f}',f'{r[3]/2**20:.3f}',f'{r[4]/1024:.3f}',f'{r[5]:.3f}',f'{r[6]/2**20:.3f}',r[7]]for r in rows]),prep_table=table(['批量ReLU数','离线批次s中位数','摊销离线μs/ReLU','离线B/ReLU(两方)','在线B/ReLU(两方)'],[[n,f"{med(f[(n,)],'offline_s'):.6f}",f"{med(f[(n,)],'offline_s')*1e6/n:.3f}",3488,8]for n in [10340,173056]]),fc_table=table(['Benchmark','N','FC在线组件和s','在线MiB','各层setup时间和s','各层setup MiB和'],[[model,degree,f"{sum(med(v,'online_s')for v in sel):.6f}",f"{sum(med(v,'online_bytes')for v in sel)/2**20:.6f}",f"{sum(med(v,'setup_s')for v in sel):.6f}",f"{sum(med(v,'setup_bytes')for v in sel)/2**20:.3f}"]for model in ['A','B']for degree in [8192,16384]for sel in [[v for k,v in l.items()if k[0]=='fc'and k[1]==model and k[2]==degree]]]),net_table=table(['模型','ReLU数','FSS离线s','FSS keys MiB','Linear组件和s','Linear在线MiB','ReLU在线s','ReLU在线MiB'],[[model,n,f"{med(f[(n,)],'offline_s'):.6f}",f"{med(f[(n,)],'offline_key_bytes')/2**20:.3f}",f"{sum(med(z,'online_s')for z in sel):.6f}",f"{sum(med(z,'online_bytes')for z in sel)/2**20:.3f}",f"{med(f[(n,)],'online_s'):.6f}",f"{med(f[(n,)],'online_bytes')/2**20:.6f}"]for model,n in [('A',10340),('B',173056)]for sel in [[v for k,v in l.items()if k[1]==model and k[2]==8192]]]),pool_table=table(['离线s中位数','离线key MiB','在线ms中位数','在线双向KiB','错误'],[[f"{med(pool,'offline_s'):.6f}",f"{med(pool,'offline_key_bytes')/2**20:.3f}",f"{med(pool,'online_s')*1000:.3f}",f"{med(pool,'online_bytes')/1024:.3f}",sum(x['errors']for x in pool)]]),acc_table=table(['模型','完整测试数','明文准确率','定点模拟准确率','HE/FSS样本数','样本预测与明文一致率','样本最大logit误差'],[[m,a['full_test_examples'],f"{a['plaintext_accuracy']*100:.2f}%",f"{a['fixedpoint_simulation_accuracy']*100:.2f}%",a['sample_count'],f"{a['sample_prediction_agreement']*100:.1f}%",f"{max(r['logits_max_abs_difference']for r in a['samples']):.6g}"]for m,a in accuracy.items()]))
(ROOT/'REPORT.md').write_text(report)
manifest={}
for folder in [ROOT.parent/'CRISP',ROOT.parent/'CKKS-master']:
 for sub in ['validated','bench','tests','test/relu']:
  d=folder/sub
  if d.exists():
   for p in d.glob('*'):
    if p.suffix in ['.cpp','.h']:manifest[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [folder/'CMakeLists.txt']:
  manifest[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (ROOT/'models').glob('*.pt'):manifest[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in list(ROOT.glob('*.py'))+[ROOT/'build.sh',ROOT/'requirements.txt',ROOT/'EXPERIMENTS.json',ROOT/'DATA_MODEL_PROVENANCE.json',ROOT.parent/'CKKS-master/emp-sh2pc/semihonest.h']:
 manifest[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'sha256-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Wrote REPORT.md, CSVs, summary.json, figures, source/checkpoint hashes')
