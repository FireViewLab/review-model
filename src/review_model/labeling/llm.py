"""검수 패킷을 LLM으로 판정한다(Anthropic Messages 호환 엔드포인트).

결과는 정답이 아니라 2차 의견이다. 작성자 관련 컬럼과 기존 라벨 칸은 보내지 않는다.
키는 `~/.config/review-v4/llm.env`의 LLM_BASE_URL과 LLM_API_KEY* 이며, 여러 키가 있으면 배치마다 번갈아 쓴다.

    python -m review_model.labeling.llm packet/v4_pilot_400.xlsx llm/pilot_opus.xlsx claude-opus-4-8 --workers 6
"""

import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from review_model.constants import ID_COLUMNS, LABELS, SIGNAL_CODES
from review_model.labeling.prompt import SYSTEM

ENV_PATH = pathlib.Path.home() / ".config/review-v4/llm.env"
# 검수자 입력 칸과 작성자 식별 정보는 모델에 보내지 않는다.
EXCLUDED = (
    "작성자",
    "author",
    "reviewer",
    "최종 라벨",
    "광고 표시",
    "확인된 신호",
    "근거 메모",
    "데이터 미확인",
    "검수자",
    "판정 시각",
    "human_label",
    "signal_types",
    "evidence_note",
    "reviewed_at",
    "calibration_order",
)


def load_env(path=ENV_PATH):
    env = dict(line.strip().split("=", 1) for line in open(path) if "=" in line)
    keys = [v for k, v in env.items() if k.startswith("LLM_API_KEY")]
    return env["LLM_BASE_URL"], keys


def records(df: pd.DataFrame, id_col: str):
    cols = [c for c in df.columns if c != id_col and not any(x in str(c) for x in EXCLUDED)]
    for _, row in df.iterrows():
        rec = {"id": str(row[id_col])}
        for c in cols:
            v = row[c]
            rec[c] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
        yield rec


def call(client, base, key, model, batch, retries=4):
    body = {
        "model": model,
        "max_tokens": 4000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": json.dumps(batch, ensure_ascii=False, default=str)}],
    }
    for attempt in range(retries):
        try:
            r = client.post(
                f"{base}/v1/messages",
                json=body,
                timeout=180,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
            return json.loads(text[text.index("[") : text.rindex("]") + 1]), data.get("usage", {})
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            if attempt == retries - 1:
                raise
            print(f"  재시도 {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))


def to_row(id_col, rec_id, answer):
    label = answer.get("label")
    return {
        id_col: rec_id,
        "llm_label": label if label in LABELS else None,
        "llm_disclosed": answer.get("disclosed"),
        "llm_signal_types": ";".join(s for s in answer.get("signals") or [] if s in SIGNAL_CODES),
        "llm_evidence_note": answer.get("note"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packet")
    ap.add_argument("out")
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)

    df = pd.read_excel(a.packet, sheet_name=0)
    id_col = next(c for c in ID_COLUMNS if c in df.columns)
    recs = list(records(df, id_col))
    base, keys = load_env()
    batches = [recs[i : i + a.batch] for i in range(0, len(recs), a.batch)]
    used = {"input_tokens": 0, "output_tokens": 0}
    done = [0]

    def work(n):
        with httpx.Client() as client:
            answers, usage = call(client, base, keys[n % len(keys)], a.model, batches[n])
        by_id = {str(x.get("id")): x for x in answers}
        rows = [to_row(id_col, rec["id"], by_id.get(rec["id"], {})) for rec in batches[n]]
        for k in used:
            used[k] += usage.get(k, 0)
        done[0] += len(rows)
        print(f"{a.model} {done[0]}/{len(recs)}", flush=True)
        return rows

    with ThreadPoolExecutor(a.workers) as pool:
        results = [row for rows in pool.map(work, range(len(batches))) for row in rows]
    pd.DataFrame(results).to_excel(a.out, index=False, sheet_name="llm")
    missing = sum(1 for r in results if not r["llm_label"])
    print(
        f"완료 {a.model} {len(results)}건 · 라벨 없음 {missing} · "
        f"토큰 입력 {used['input_tokens']:,} 출력 {used['output_tokens']:,}"
    )


if __name__ == "__main__":
    main()
