"""보고서의 절 순서대로 전 단계를 실행한다.

실행: python src/run_all.py          분류 모델을 빼고 실행 (몇 분)
      python src/run_all.py --model  분류 모델까지 실행 (한 시간 안팎)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["step1_split", "step2_missing", "step3_person_genre", "step5_dpi", "step6_compare", "step7_improve",
         "step8_region_venue"]
if "--model" in sys.argv:
    STEPS.insert(3, "step4_genre_gcn")
    STEPS.append("step9_confident_pairs")      # 분류 모델의 예측이 있어야 한다

for s in STEPS:
    print(f"\n{'=' * 20} {s} {'=' * 20}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / f"{s}.py")])
    if r.returncode != 0:
        sys.exit(f"{s} 에서 멈춤")

import pandas as pd  # noqa: E402

checks = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in sorted((HERE.parent / "outputs" / "checks").glob("step*.csv"))])
checks.to_csv(HERE.parent / "outputs" / "checks" / "_all.csv", index=False, encoding="utf-8-sig")
print("\n기준값과 비교:", checks["일치"].value_counts().to_dict())
