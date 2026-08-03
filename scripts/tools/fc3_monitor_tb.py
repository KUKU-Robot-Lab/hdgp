"""fc3(lstm_test3, 148종) 모니터링 — 현재 상태 + fc2(lstm_test2, 154종) 비교.

사용: isaaclab.sh -p fc3_monitor_tb.py <events_dir> <right|left>
fc2 최종 기준값은 아래 FC2_BASELINE 에 내장(2026-07-16 완주분).
붕괴 신호(NaN/Inf, rewards/step 폭락)와 per-object 성공률을 덤프한다.
"""
import sys, math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# fc2 (lstm_test2, 154종, 완주) 최종 기준
FC2_BASELINE = {
    "right": dict(adr=50, epoch=20000, median=0.795, mean=0.761, lt05=5, lt005=3,
                  cup=0.04, rewards_step=927, in_success=0.359),
    "left":  dict(adr=31, epoch=20000, median=0.690, mean=0.648, lt05=11, lt005=10,
                  cup=0.02, rewards_step=1243, in_success=0.346),
}

path, side = sys.argv[1], sys.argv[2]
base = FC2_BASELINE.get(side, {})
ea = EventAccumulator(path, size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags().get('scalars', [])

def S(tag): return ea.Scalars(tag) if tag in tags else []

print(f"\n########## fc3 {side.upper()} (lstm_test3, 148종) ##########")
# epoch / adr
adr = S('num_adr_increases/iter')
cur_epoch = adr[-1].step if adr else (S('rewards/step')[-1].step if S('rewards/step') else 0)
cur_adr = adr[-1].value if adr else float('nan')
print(f"현재 epoch~{cur_epoch}  ADR={cur_adr:.0f}   [fc2 최종: epoch {base.get('epoch')}, ADR {base.get('adr')}]")

# rewards/step + 붕괴 체크
rs = S('rewards/step')
if rs:
    vals = [e.value for e in rs[-20:]]
    nan = any(math.isnan(v) or math.isinf(v) for v in vals)
    print(f"rewards/step 최신={rs[-1].value:.1f}  최근20 min={min(vals):.1f}/max={max(vals):.1f}  NaN/Inf={nan}   [fc2 최종 {base.get('rewards_step')}]")
    if nan: print("  *** 붕괴 경보: NaN/Inf 검출 ***")
insucc = S('in_success_region/iter')
if insucc:
    print(f"in_success_region 최신={insucc[-1].value:.3f}   [fc2 최종 {base.get('in_success')}]")

# 안정성
for t,label in [('losses/entropy','entropy'),('info/kl','kl'),('losses/c_loss','c_loss')]:
    ev = S(t)
    if ev: print(f"{label}={ev[-1].value:.4f}", end="  ")
print()

# per-object
obj = [t for t in tags if t.startswith('episode_success_rate/')]
finals = []
cup_val = None
for t in obj:
    ev = S(t)
    if ev:
        nm = t.replace('episode_success_rate/','').replace('/iter','')
        finals.append((nm, ev[-1].value))
        if nm == 'cup': cup_val = ev[-1].value
        if nm == 'cup_big': cup_big_val = ev[-1].value
if finals:
    vs = sorted(v for _,v in finals); n=len(vs)
    median=vs[n//2]; mean=sum(vs)/n
    lt05=sum(1 for v in vs if v<0.5); lt005=sum(1 for v in vs if v<0.05)
    print(f"per-object({n}종): 중앙값={median:.3f} 평균={mean:.3f} <0.5:{lt05} <0.05:{lt005}   "
          f"[fc2: 중앙값 {base.get('median')} 평균 {base.get('mean')} <0.5:{base.get('lt05')} <0.05:{base.get('lt005')}]")
    fs = sorted(finals, key=lambda x:x[1])
    print("  하위8: " + ", ".join(f"{nm}={v:.2f}" for nm,v in fs[:8]))
    # ★핵심 변경검증: cup (fc2 자산버그로 0.04였음 → 수정 후 상승해야)
    cv = next((v for nm,v in finals if nm=='cup'), None)
    cbv = next((v for nm,v in finals if nm=='cup_big'), None)
    print(f"  ★cup={cv if cv is None else round(cv,3)} (fc2 {base.get('cup')} ← 자산버그)  cup_big={cbv if cbv is None else round(cbv,3)} (신규)")
