"""검수 패킷에 LLM 2차 의견(제안 라벨·신호·근거)을 붙인다. 정답이 아니라 다시 볼 곳을 찾는 용도다.

사용:
  python llm_opinion.py <packet.xlsx> <out.xlsx> <model> [--batch 20] [--workers 6]

키는 ~/.config/review-v4/llm.env 의 LLM_API_KEY, LLM_API_KEY_2 를 배치마다 번갈아 쓴다(사용량 반반).
작성자 해시·작성자 관련 컬럼은 보내지 않는다.
"""

import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

LABELS = ["NORMAL", "SUSPICIOUS", "DISCLOSED_PROMO", "UNCERTAIN", "INVALID"]
SIGNALS = [
    "TEXT_PRODUCT_DESCRIPTION", "TEXT_UNTRIED_CLAIM", "TEXT_PROMOTIONAL", "TEXT_OVERCONFIDENT", "TEXT_REPEAT",
    "LOW_INFO_COMBINED", "DATE_CONCENTRATION", "HOUR_CONCENTRATION", "RATING_SKEW", "HELPFUL_ANOMALY",
    "CROSS_PRODUCT_EXACT_DUPLICATE", "SAME_PRODUCT_EXACT_DUPLICATE", "AUTHOR_REPEAT", "DISCLOSED_PROMO",
]
ID_COLS = ("항목ID", "annotation_id", "확인ID")
DROP = ("작성자", "author", "reviewer", "최종 라벨", "광고 표시", "확인된 신호", "근거 메모", "데이터 미확인", "검수자",
        "판정 시각", "human_label", "signal_types", "evidence_note", "reviewed_at", "calibration_order")

SYSTEM = f"""너는 한국 쇼핑몰 리뷰 신뢰도 라벨링의 2차 검수자다. Re:view v4.0 라벨링 기준으로 각 리뷰를 독립적으로 판정한다.

라벨:
- NORMAL: 원문과 관련 데이터를 검토했고 정의된 의심 신호가 명확하게 없음. 사진·긴 본문만으로 자동 확정하지 않는다.
- SUSPICIOUS: 텍스트·작성 행동·리뷰 간 관계에서 구체적 위험 근거가 관찰됨. 단순 칭찬이나 일반 추천만으로는 의심으로 확정하지 않는다. 신호 하나로 판정하면 근거와 반대 가능성을 메모에 쓴다.
- DISCLOSED_PROMO: 체험단·협찬·광고·제품 제공을 명시적으로 공개함. 의심 신호가 함께 있어도 최종 라벨은 DISCLOSED_PROMO이고 의심 신호는 signals에 같이 적는다.
- UNCERTAIN: 정보 부족, 상충하는 근거, 정상적 대안 설명이 남아 있음. 근거 부족을 정상·의심의 증거로 쓰지 않는다.
- INVALID: 리뷰 본문이 아니거나 손상·오입력으로 검수 불가.

텍스트 신호: 상품 설명문 형태(TEXT_PRODUCT_DESCRIPTION), 미사용 상태의 효과 단정(TEXT_UNTRIED_CLAIM), 홍보성·판매 유도가 본문을 지배(TEXT_PROMOTIONAL), 모든 사용자에게 같은 효과를 보장하는 과도한 확신(TEXT_OVERCONFIDENT), 리뷰 내부 비정상 반복(TEXT_REPEAT).
복합·행동·관계 신호: 사진 0장(확인)·5점·의미 문자 10자 미만(LOW_INFO_COMBINED, 우선 검토 대상이지 자동 의심 아님), 같은 상품 하루/한 시간 몰림(DATE_CONCENTRATION/HOUR_CONCENTRATION), 상품 별점 분포 편차(RATING_SKEW), 도움 수 특이(HELPFUL_ANOMALY), 다른 상품/같은 상품 동일 문장(CROSS_PRODUCT_EXACT_DUPLICATE/SAME_PRODUCT_EXACT_DUPLICATE), 작성자 반복(AUTHOR_REPEAT), 광고 표시(DISCLOSED_PROMO).
수치 해석:
- 동일 문장 수에는 "좋아요 잘 쓸게요" 같은 짧은 상투 문구나, 같은 리뷰가 옵션 상품 여러 개에 공유되어 잡힌 경우가 섞여 있다. 본문이 구체적이고 길며(의미 문자 30자 이상) 상투적이지 않은 문장이 다른 상품에 반복될 때만 중복 근거로 쓴다. 숫자만으로 SUSPICIOUS를 주지 않는다.
- 별점 분포는 플랫폼 평소(5점 77~96%)와 비교한다. 그 범위 안이면 RATING_SKEW가 아니다.
- 몰림 수치는 그 상품의 평소 하루 리뷰 수보다 크게 많을 때만 근거가 된다.
- 행동·관계 신호 하나만 있고 본문이 평범한 사용 후기면 SUSPICIOUS보다 UNCERTAIN을 우선 검토한다.
- 짧은 만족 표현(예: 좋아요, 잘 쓸게요)만 있고 다른 신호가 없으면 NORMAL이다. 짧다는 이유만으로 UNCERTAIN을 주지 않는다.
정보 상태: 사진 "미확인"은 0장이 아니다. 별점이 없으면 5점으로 가정하지 않는다. 플랫폼 대부분 5점 비율이 80~95%라 5점 자체는 신호가 아니다.

출력: 다른 말 없이 JSON 배열만. 입력 순서대로 각 원소는
{{"id": "<항목ID>", "label": one of {LABELS}, "disclosed": "Y"|"N", "signals": [허용 코드: {SIGNALS}], "note": "한국어 한두 문장 근거"}}"""


def load_env():
    env = {}
    for line in open(pathlib.Path.home() / ".config/review-v4/llm.env"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            env[k] = v
    return env["LLM_BASE_URL"], [v for k, v in env.items() if k.startswith("LLM_API_KEY")]


def items(df):
    idc = next(c for c in ID_COLS if c in df.columns)
    cols = [c for c in df.columns if c != idc and not any(d in str(c) for d in DROP)]
    for _, r in df.iterrows():
        rec = {"id": str(r[idc])}
        rec.update({c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c])) for c in cols})
        yield rec


def call(client, base, key, model, batch):
    body = {
        "model": model,
        "max_tokens": 4000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": json.dumps(batch, ensure_ascii=False, default=str)}],
    }
    for attempt in range(4):
        try:
            r = client.post(f"{base}/v1/messages", json=body, timeout=180,
                            headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
            r.raise_for_status()
            d = r.json()
            text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text").strip()
            text = text[text.index("["): text.rindex("]") + 1]
            return json.loads(text), d.get("usage", {})
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            if attempt == 3:
                raise
            print(f"  재시도 {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_excel(a.src, sheet_name=0)
    idc = next(c for c in ID_COLS if c in df.columns)
    recs = list(items(df))
    base, keys = load_env()
    batches = [recs[i: i + a.batch] for i in range(0, len(recs), a.batch)]
    used = {"input_tokens": 0, "output_tokens": 0}
    done = [0]

    def work(n):
        with httpx.Client() as client:
            got, usage = call(client, base, keys[n % len(keys)], a.model, batches[n])
        by = {str(g.get("id")): g for g in got}
        rows = []
        for rec in batches[n]:
            g = by.get(rec["id"], {})
            rows.append({
                idc: rec["id"],
                "llm_label": g.get("label") if g.get("label") in LABELS else None,
                "llm_disclosed": g.get("disclosed"),
                "llm_signal_types": ";".join(s for s in g.get("signals") or [] if s in SIGNALS),
                "llm_evidence_note": g.get("note"),
            })
        for k in used:
            used[k] += usage.get(k, 0)
        done[0] += len(rows)
        print(f"{a.model} {done[0]}/{len(recs)}", flush=True)
        return rows

    with ThreadPoolExecutor(a.workers) as pool:
        results = [r for rows in pool.map(work, range(len(batches))) for r in rows]
    pd.DataFrame(results).to_excel(a.out, index=False, sheet_name="llm")
    missing = sum(1 for r in results if not r["llm_label"])
    print(f"완료 {a.model} {len(results)}건 · 라벨 없음 {missing} · 토큰 입력 {used['input_tokens']:,} 출력 {used['output_tokens']:,}")


if __name__ == "__main__":
    main()
