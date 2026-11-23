#!/usr/bin/env python3
import json,collections,time
from common import *
tasks=list((RUN/'tasks').glob('*.json'));completed={p.stem for p in (RUN/'scores').glob('*.json')};done=[p for p in tasks if p.stem in completed]
state={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'tasks':len(tasks),'completed':len(done),'queue_closed':(RUN/'tasks/CLOSED').exists(),
 'by_stage_total':dict(collections.Counter(p.stem.split('.')[0] for p in tasks)),'by_stage_completed':dict(collections.Counter(p.stem.split('.')[0] for p in done)),
 'worker_errors':[str(p) for p in (RUN/'logs/workers').glob('*.error.json') if json.loads(p.read_text()).get('task') not in completed],
 'final_analysis_complete':(RUN/'FINAL_COMPLETE.json').exists()}
print(json.dumps(state,indent=2));save_json(state,RUN/'status.json')
