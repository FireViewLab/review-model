"""시험 세트 일부를 사람이 확인할 수 있게 패킷으로 뽑는다. LLM 라벨은 숨기고 key 파일에만 남긴다.

사용: python verify_packet.py <split_dir>/test.csv <out_dir> [--normal 58]
뽑는 것: 시험 세트의 의심 전부 + 정상 무작위 N건. 화면 컬럼은 검수 패킷과 같다.
"""

import argparse
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKETS = [("packet/v4_pilot_400.xlsx", "packet/v4_pilot_key.csv"),
           ("packet/v4_main_5000.xlsx", "packet/v4_main_5000_key.csv")]
LABEL_COLS = ["최종 라벨", "광고 표시(Y/N)", "확인된 신호 코드", "근거 메모", "데이터 미확인 항목", "검수자", "판정 시각"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--normal", type=int, default=58)
    a = ap.parse_args()
    test = pd.read_csv(a.test, dtype=str)
    pick = pd.concat([test[test.y == "1"], test[test.y == "0"].sample(a.normal, random_state=20261122)])
    frames = []
    for packet, key in PACKETS:
        p = pd.read_excel(ROOT / packet, dtype=str)
        k = pd.read_csv(ROOT / key, dtype=str)[["item_id", "platform", "review_id"]]
        frames.append(p.merge(k, left_on="항목ID", right_on="item_id"))
    allp = pd.concat(frames, ignore_index=True).drop_duplicates(["platform", "review_id"], keep="last")
    sel = allp.merge(pick[["platform", "review_id", "y"]], on=["platform", "review_id"])
    sel = sel.sample(frac=1, random_state=7).reset_index(drop=True)
    sel["확인ID"] = [f"V{i + 1:03d}" for i in range(len(sel))]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    view = sel.drop(columns=["item_id", "platform", "review_id", "y", "항목ID"])
    view = view[["확인ID"] + [c for c in view.columns if c not in LABEL_COLS + ["확인ID"]] + LABEL_COLS]
    for c in LABEL_COLS:
        view[c] = ""
    view.to_excel(out / "v4_test_verify_150.xlsx", index=False)
    sel[["확인ID", "항목ID", "platform", "review_id", "y"]].rename(columns={"y": "llm_y"}).to_csv(
        out / "v4_test_verify_key.csv", index=False)
    print(f"{len(sel)}건 (LLM 의심 {int((sel.y == '1').sum())}, 정상 {int((sel.y == '0').sum())}) → {out}")


if __name__ == "__main__":
    main()
