"""Small live progress message; no expensive model or dataframe imports."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
if (root/'FINAL_COMPLETE.json').exists():
 print('FINAL_READY '+(root/'FINAL_COMPLETE.json').read_text());raise SystemExit
p=root/'watcher_status.json'
s=json.loads(p.read_text())
if s.get('errors'):print('CHECK_ERRORS '+json.dumps(s,ensure_ascii=False))
elif s['done']==s['tasks']:
 log=(root.parent/'finalize.log').read_text(errors='replace')
 print('CHECK_FINALIZER_ERROR '+log[-2000:] if 'Traceback (most recent call last)' in log else '全部预测已完成，正在核验配对、计算统计并导出最终图。')
else:print(f"TF 已完成；promoter 已完成 {s['done']-1542:,}/{s['tasks']-1542:,} 个任务，当前未处理错误 0。")
