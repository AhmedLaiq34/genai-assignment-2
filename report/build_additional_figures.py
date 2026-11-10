from pathlib import Path
import json,csv
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('.');G=R/'report/generated';F=R/'report/figures';F.mkdir(exist_ok=True)
cache={v:json.loads((R/f'results/cache/benchmark{("_adapter" if v=="adapted" else "")}.json').read_text()) for v in ['base','adapted']}
ann_ids=sorted({r['annotation_id'] for d in cache.values() for r in d['records']})
labels=[];vals=[];mems=[];source=[]
for ann in ann_ids:
 for variant,d in cache.items():
  for mode in ['uncached','cached_explicit']:
   r=next(x for x in d['records'] if x['annotation_id']==ann and x['mode']==mode)
   latency=r['latency_seconds']['median']; peaks=sorted(x['peak_allocated_bytes'] for x in r['repetitions']); memory=peaks[len(peaks)//2]/(1024**3)
   labels.append(f'{variant}\n{mode.replace("_", " ")}'); vals.append(latency);mems.append(memory);source.append([variant,ann,r['prompt_tokens'],mode,latency,memory,r['generated_tokens']])
with (G/'cache_plot_source.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['variant','annotation_id','prompt_tokens','mode','median_latency_s','median_peak_allocated_gib','generated_tokens']);w.writerows(source)
fig,(ax,bx)=plt.subplots(2,1,figsize=(6.4,4.1),sharex=True);x=range(len(vals));colors=['#79a9c4' if 'uncached' in z else '#245f7a' for z in labels];ax.bar(x,vals,color=colors);ax.set_ylabel('Median seconds');ax.set_title('Explicit KV-cache benchmark (two validation prompts; five runs)');ax.grid(axis='y',alpha=.2);bx.bar(x,mems,color=colors);bx.set_ylabel('Peak allocated GiB');bx.set_xticks(list(x),labels,fontsize=7);bx.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(F/'cache_benchmarks.pdf');plt.close(fig)
# Speculative memory and time from adapter-backed paired benchmark.
spec=json.loads((R/'results/speculative/benchmark_adapter.json').read_text()); rows=spec['rows']; names=[];times=[];memory=[];src=[]
for q in rows:
 name='Target cached' if q['mode']=='cached' else f'Spec k={q["draft_k"]}'; names.append(name);times.append(q['median_s']);memory.append(q['peak_allocated_bytes']/(1024**3));src.append([name,q['prompt_tokens'],q['cap'],q['median_s'],min(q['times_s']),max(q['times_s']),memory[-1],q['token_match']])
with (G/'speculative_plot_source.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['mode','prompt_tokens','max_new_tokens','median_s','min_s','max_s','peak_allocated_gib','token_match']);w.writerows(src)
fig,(ax,bx)=plt.subplots(2,1,figsize=(5.8,4.0),sharex=True);ax.bar(range(len(rows)),times,color=['#245f7a' if q['mode']=='cached' else '#a9bdc8' for q in rows]);ax.set_ylabel('Median seconds');ax.set_title('Speculative decoding, adapter target');ax.grid(axis='y',alpha=.2);bx.bar(range(len(rows)),memory,color=['#245f7a' if q['mode']=='cached' else '#a9bdc8' for q in rows]);bx.set_ylabel('Peak allocated GiB');bx.set_xticks(range(len(rows)),names);bx.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(F/'speculative_memory.pdf');plt.close(fig)
# Attention tile error and memory (time plot exists already).
at=json.loads((R/'results/attention/benchmark.json').read_text()); rows=at['rows'];src=[]
for q in rows:
 src.append([q['length'],q['tile'],q['error']['max_abs_error'],q['dense']['peak_incremental_bytes']/2**20,q['tiled']['peak_incremental_bytes']/2**20])
with (G/'attention_error_memory_source.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['length','tile','max_abs_error','dense_incremental_peak_mib','tiled_incremental_peak_mib']);w.writerows(src)
fig,ax=plt.subplots(figsize=(5.7,2.7))
for t in [16,32,64]:
 z=[q for q in rows if q['tile']==t];ax.plot([q['length'] for q in z],[q['error']['max_abs_error'] for q in z],marker='o',label=f'Tile {t}')
ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set(xlabel='Sequence length',ylabel='Maximum absolute error',title='Tiled versus dense FP32 attention');ax.legend(frameon=False,ncol=3);ax.grid(which='both',alpha=.2);fig.tight_layout();fig.savefig(F/'attention_error.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(5.7,2.7));lens=sorted({q['length'] for q in rows});dense=[]
for n in lens:dense.append(next(q['dense']['peak_incremental_bytes']/2**20 for q in rows if q['length']==n))
ax.plot(lens,dense,marker='o',label='Dense')
for t in [16,32,64]:ax.plot(lens,[next(q['tiled']['peak_incremental_bytes']/2**20 for q in rows if q['length']==n and q['tile']==t) for n in lens],marker='o',label=f'Tiled {t}')
ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set(xlabel='Sequence length',ylabel='Incremental peak allocation (MiB)',title='Attention temporary memory');ax.legend(frameon=False,ncol=2,fontsize=7);ax.grid(which='both',alpha=.2);fig.tight_layout();fig.savefig(F/'attention_memory.pdf');plt.close(fig)
