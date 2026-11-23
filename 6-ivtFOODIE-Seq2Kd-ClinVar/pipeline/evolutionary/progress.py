#!/usr/bin/env python3
import json,time
from shared import *
for name,path in [('Seq2Kd full queue',RUN/'watcher_status.json'),('ESM masked queue',OUT/'status.json')]:
 d=json.loads(path.read_text());print(name,f'{d["done"]:,}/{d["tasks"]:,}',f'status age={time.time()-d["time"]:.0f}s',('errors='+str(d.get('errors',[]))) if 'errors' in d else '')
for name,p in [('AM-only',OUT/'qc/AM_BRANCH_COMPLETE.json'),('Evolutionary full',OUT/'COMPLETE.json'),('Original full',RUN/'FINAL_COMPLETE.json')]:print(name,'COMPLETE' if p.exists() else 'running')
for f in (OUT/'logs').glob('*.log'):
 tail=f.read_text()[-3000:]
 if 'Traceback (most recent call last)' in tail:print('CHECK WORKER ERROR',f)
