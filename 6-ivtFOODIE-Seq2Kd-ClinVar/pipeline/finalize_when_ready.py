#!/usr/bin/env python3
"""Durable watcher: only complete and verified predictions may produce final figures."""
import time,subprocess,sys,fcntl
from common import *
watcher_lock=open(RUN/'finalize.lock','a')
try:fcntl.flock(watcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('A finalizer is already running')
while True:
 tasks=list((RUN/'tasks').glob('*.json'))
 completed={p.stem for p in (RUN/'scores').glob('*.json')}
 done=sum(p.stem in completed for p in tasks)
 errors=[p for p in (RUN/'logs/workers').glob('*.error.json') if json.loads(p.read_text()).get('task') not in completed]
 save_json({'tasks':len(tasks),'done':done,'errors':[str(p) for p in errors],'time':time.time()},RUN/'watcher_status.json')
 if (RUN/'tasks/CLOSED').exists() and tasks and done==len(tasks):break
 time.sleep(30)
subprocess.run([sys.executable,'-u',str(ROOT/'analyze.py')],check=True,cwd=ROOT)
