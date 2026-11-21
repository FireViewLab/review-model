"""모델을 실제 리뷰 test 세트와 프로브 문장으로 평가해 관문 통과 여부를 낸다.

사용:
  python eval_v4.py <model_dir> <split_dir>/test.csv <out.json> [--baseline <v3.1 model_dir>]

관문 (v4.0 로드맵 5-3):
- test PR-AUC가 기준 모델(v3.1)보다 높다.
- 정밀도 목표 지점의 재현율을 같이 보고한다.
- 프로브: 참고용으로만 보고한다(결정 ⑥).
- 플랫폼별 PR-AUC 편차를 보고한다(양성이 5건 이상인 플랫폼만).
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_v4 import average_precision, load, logits_of, pr_points

# 광고를 밝힌 문장(협찬·체험단)은 결정 ③에 따라 SUSPICIOUS가 아니라 DISCLOSED_PROMO라서
# 의심 프로브에서 빼고 따로 보고만 한다(관문에 넣지 않음).
PROBE_SUSPICIOUS = [
    "최고최고 강추 무조건 사세요 진짜 대박이에요 ㅋㅋㅋ", "좋아요 좋아요 좋아요 좋아요 좋아요 좋아요",
    "이 가격에 이 퀄리티 실화? 인생템 등극!! 다들 사세요!!!", "리뷰 이벤트 참여합니다 별5개 드려요",
    "대박 대박 대박 무조건 재구매 각 최고의 선택", "사장님 번창하세요 최고의 제품 최고의 서비스 최고",
]
PROBE_DISCLOSED = [
    "협찬받고 작성한 리뷰입니다 너무 좋아요 추천합니다", "체험단으로 제공받아 작성합니다. 정말 만족스러워요. 강력 추천!",
]
PROBE_GENUINE = [
    "사이즈는 정사이즈인데 어깨가 조금 좁게 나왔어요. 두 번 세탁했는데 보풀은 아직 없고 색 빠짐도 없어요.",
    "배송은 이틀 걸렸고 포장은 꼼꼼했어요. 다만 설명서가 영어로만 되어 있어서 조립할 때 좀 헤맸습니다.",
    "향이 생각보다 강해서 처음엔 별로였는데 일주일 쓰니 익숙해졌어요. 건성 피부라 저녁에만 바르고 있어요.",
    "가격 대비 괜찮은데 마감이 아쉬워요. 모서리 실밥이 하나 튀어나와 있어서 잘라냈습니다.",
    "아이가 잘 먹어서 두 번째 구매예요. 유통기한이 지난번보다 짧게 와서 별 하나 뺐어요.",
    "무게가 꽤 있어서 들고 다니기엔 불편하고 집에서 쓰기엔 충분합니다. 소음은 냉장고보다 조금 큰 정도예요.",
    "색상이 사진보다 어둡게 왔어요. 교환하려다 그냥 쓰기로 했는데 재질은 부드럽고 따뜻합니다.",
    "충전 케이블이 짧아서 따로 샀어요. 배터리는 하루 반 정도 가고 블루투스 연결은 끊김 없었어요.",
]


def score(model_dir, texts, device):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    cfg = json.load(open(f"{model_dir}/inference_config.json"))
    d = logits_of(model, tok, texts, device, cfg.get("max_length", 256))
    t = cfg.get("calibration_temperature", 1.0) if cfg.get("model_family") == "ptext-koelectra-v4.0" else 1.0
    return 1 / (1 + np.exp(-d / t)), cfg


def report(model_dir, test, device):
    p, cfg = score(model_dir, test.content, device)
    y = test.y
    th = cfg.get("threshold", 0.5)
    pred = p >= th
    tp = int((pred & (y == 1)).sum())
    per = {}
    plats = np.array(test.platform)
    for pf in sorted(set(plats)):
        m = plats == pf
        if y[m].sum() >= 5:
            per[pf] = round(average_precision(y[m], p[m]), 4)
    probe_s, _ = score(model_dir, PROBE_SUSPICIOUS, device)
    probe_g, _ = score(model_dir, PROBE_GENUINE, device)
    probe_d, _ = score(model_dir, PROBE_DISCLOSED, device)
    _, prec, rec = pr_points(y, p)
    return {
        "pr_auc": round(average_precision(y, p), 4),
        "threshold": th,
        "precision": round(tp / max(pred.sum(), 1), 4),
        "recall": round(tp / max((y == 1).sum(), 1), 4),
        "recall_at_precision_0.8": round(float(rec[prec >= 0.8].max()) if (prec >= 0.8).any() else 0.0, 4),
        "per_platform_pr_auc": per,
        "per_platform_spread": round(max(per.values()) - min(per.values()), 4) if len(per) > 1 else None,
        "probe_suspicious_mean": round(float(probe_s.mean()), 4),
        "probe_genuine_mean": round(float(probe_g.mean()), 4),
        "probe_gap": round(float(probe_s.mean() - probe_g.mean()), 4),
        "probe_disclosed_mean": round(float(probe_d.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = pathlib.Path(a.test)
    test = load(path.parent, path.stem)
    result = {"n": len(test), "suspicious": int(test.y.sum()), "candidate": report(a.model, test, device)}
    if a.baseline:
        result["baseline"] = report(a.baseline, test, device)
    c, b = result["candidate"], result.get("baseline")
    result["gate"] = {
        "pr_auc_above_baseline": None if not b else c["pr_auc"] > b["pr_auc"],
        # 결정 ⑥(2026-11-22): 칭찬·홍보 문구만 있는 짧은 리뷰는 의심이 아니다. 프로브는 참고용이며 관문이 아니다.
        "probe_gap_reference_only": c["probe_gap"],
        "platform_spread": c["per_platform_spread"],
    }
    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
