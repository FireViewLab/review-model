"""후보 풀에서 시범 검수 패킷을 만든다.

검수자에게는 후보 층(stratum)과 신호 플래그를 숨기고, 판단에 필요한 원문·관련 데이터만 준다.
층·플래그·원본 식별자는 별도 키 파일에만 남긴다.
"""
import csv
import json
import hashlib
import pathlib
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

POOL = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
PRODUCTS = pathlib.Path(sys.argv[3])
CTX = pathlib.Path(sys.argv[4])
# 본 검수: 선택 목록(platform, review_id, stratum)과 출력 이름·항목ID 접두어를 받는다.
SELECTION = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else None
NAME = sys.argv[6] if len(sys.argv) > 6 else "v4_pilot_400"
PREFIX = sys.argv[7] if len(sys.argv) > 7 else "P"
OUT.mkdir(parents=True, exist_ok=True)
csv.field_size_limit(10**8)

NAMES = {"oliveyoung": "올리브영", "kurly": "컬리", "elevenst": "11번가", "musinsa": "무신사", "ohouse": "오늘의집", "naver": "네이버"}
# 시범 검수 400건: 큰 플랫폼 80건, 작은 플랫폼 40건. 그중 60%는 신호 후보, 40%는 무작위 대조군.
QUOTA = {"oliveyoung": 80, "kurly": 80, "elevenst": 80, "musinsa": 80, "ohouse": 40, "naver": 40}
CAND_SHARE = 0.6
STRATA = ["combo", "promo", "unused", "certain", "repeat", "spec", "disclosed", "burst", "network", "author"]
LABELS = ["NORMAL", "SUSPICIOUS", "DISCLOSED_PROMO", "UNCERTAIN", "INVALID"]
# 팀 시범 검수(60건)에서 쓴 코드와 맞춘다. 여러 개는 세미콜론(;)으로 구분한다.
SIGNALS = [
    ("TEXT_PRODUCT_DESCRIPTION", "상품 설명문 형태"), ("TEXT_UNTRIED_CLAIM", "미사용 상태의 효과 단정"),
    ("TEXT_PROMOTIONAL", "홍보성·판매 유도"), ("TEXT_OVERCONFIDENT", "과도한 효과 확신"),
    ("TEXT_REPEAT", "리뷰 내부 비정상적 반복"), ("LOW_INFO_COMBINED", "사진 0장(확인)·5점·의미 문자 10자 미만"),
    ("DATE_CONCENTRATION", "같은 상품 하루 몰림"), ("HOUR_CONCENTRATION", "같은 상품 한 시간 몰림 (컬리·무신사·네이버)"),
    ("RATING_SKEW", "상품 별점 분포가 플랫폼 평소와 크게 다름"), ("HELPFUL_ANOMALY", "도움돼요 수 특이"),
    ("CROSS_PRODUCT_EXACT_DUPLICATE", "다른 상품에 같은 문장"), ("SAME_PRODUCT_EXACT_DUPLICATE", "같은 상품에 같은 문장"),
    ("AUTHOR_REPEAT", "같은 작성자 반복 작성"), ("DISCLOSED_PROMO", "광고·협찬·제공 표시"),
]
PRECISION = {"kurly": "초 단위", "musinsa": "초 단위", "naver": "초 단위"}
rng = random.Random(20261121)

products = {}
with PRODUCTS.open() as fh:
    for r in csv.DictReader(fh):
        products[(r["platform"], r["product_id"])] = r
ctx = {}
with CTX.open() as fh:
    for r in csv.DictReader(fh):
        ctx[(r["platform"], r["product_id"], r["review_id"])] = r


def pick(rows, n):
    """층마다 돌아가며 한 건씩 뽑아 특정 신호가 패킷을 독점하지 않게 한다."""
    by = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
    for v in by.values():
        rng.shuffle(v)
    chosen, seen = [], set()
    while len(chosen) < n and any(by.values()):
        for s in STRATA:
            while by[s]:
                r = by[s].pop()
                if r["review_id"] not in seen:
                    seen.add(r["review_id"])
                    chosen.append(r)
                    break
            if len(chosen) >= n:
                break
    return chosen, seen


items = []
for pf, total in ({} if SELECTION else QUOTA).items():
    path = POOL / f"{pf}.csv"
    if not path.exists():
        continue
    rows = list(csv.DictReader(path.open()))
    cands = [r for r in rows if r["stratum"] != "random"]
    cand, seen = pick(cands, round(total * CAND_SHARE))
    rand = [r for r in rows if r["stratum"] == "random" and r["review_id"] not in seen]
    rng.shuffle(rand)
    items += cand + rand[: total - len(cand)]
if SELECTION:
    wanted = {(r["platform"], r["review_id"]): r["stratum"] for r in csv.DictReader(SELECTION.open())}
    seen = set()
    for f in sorted(POOL.glob("*.csv")):
        if f.name.endswith("_counts.csv"):
            continue
        for r in csv.DictReader(f.open()):
            k = (r["platform"], r["review_id"])
            if k in wanted and k not in seen and r["stratum"] == wanted[k]:
                seen.add(k)
                items.append(r)
rng.shuffle(items)


def image_status(r):
    n = int(r["imgs"] or 0)
    if n > 0:
        return f"{n}장"
    return "0장(확인)" if r["platform"] == "kurly" else "미확인"


def rating(r):
    if not r["rating"]:
        return "플랫폼 미제공" if r["platform"] == "kurly" else "없음"
    return f"{float(r['rating']):g}/5"


def author(r):
    h = r["author_hash"]
    if not h:
        return "미제공", ""
    if h.startswith("masked-"):
        return h, "마스킹된 닉네임이라 반복 집계 불가"
    return h, f"{r['author_n'] or '-'}건 / 상품 {r['author_prods'] or '-'}개"


COLS = [
    ("항목ID", 10), ("플랫폼", 9), ("상품명", 28), ("카테고리", 14), ("리뷰 본문", 60), ("별점", 9), ("사진", 9),
    ("작성일", 11), ("시각 정밀도", 9), ("옵션", 16), ("도움돼요", 8), ("의미 문자 수", 8),
    ("같은 상품·같은 날 리뷰", 10), ("같은 상품·같은 시간 리뷰", 10), ("상품 하루 평균 리뷰", 10),
    ("상품 전체 리뷰", 9), ("상품 별점 분포", 18), ("상품 5점 비율", 8), ("같은 상품 동일 문장", 9),
    ("다른 상품 동일 문장(리뷰 수/상품 수)", 14), ("작성자(해시)", 18), ("작성자 리뷰·상품 수", 18),
    ("최종 라벨", 16), ("광고 표시(Y/N)", 9), ("확인된 신호 코드", 14), ("근거 메모", 40), ("데이터 미확인 항목", 18),
    ("검수자", 10), ("판정 시각", 14),
]
LABEL_START = 23


def sheet_rows():
    for i, r in enumerate(items, 1):
        p = products.get((r["platform"], r["product_id"]), {})
        a_hash, a_info = author(r)
        dup = f"{r['dup_ids']} / {r['dup_prods']}" if int(r["dup_ids"] or 0) >= 2 else "없음"
        c = ctx.get((r["platform"], r["product_id"], r["review_id"]), {})
        dist = json.loads(c["rating_dist"]) if c.get("rating_dist") else {}
        total = sum(dist.values())
        five = f"{dist.get('5', 0) / total:.0%}" if total else "-"
        dist_text = " ".join(f"{k}점 {v}" for k, v in sorted(dist.items(), reverse=True)) if total else "별점 없음"
        hour = c.get("hour_c") or "시각 없음"
        yield f"{PREFIX}{i:04d}", r, [
            f"{PREFIX}{i:04d}", NAMES[r["platform"]], p.get("name", ""), p.get("category", ""), r["content"], rating(r),
            image_status(r), r["written_date"], PRECISION.get(r["platform"], "날짜만"), r["option"],
            r["helpful_count"], int(r["mlen"] or 0), int(r["day_c"] or 0), hour, float(r["day_mean"] or 0),
            int(r["prod_total"] or 0), dist_text, five, int(c.get("same_prod_dup") or 0), dup, a_hash, a_info,
        ]


def build(path, reviewer):
    wb = Workbook()
    ws = wb.active
    ws.title = "검수"
    head_fill, label_fill = PatternFill("solid", fgColor="1F2937"), PatternFill("solid", fgColor="1D4ED8")
    for c, (name, width) in enumerate(COLS, 1):
        cell = ws.cell(1, c, name)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = label_fill if c >= LABEL_START else head_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 42
    for n, (_, _, vals) in enumerate(sheet_rows(), 2):
        for c, v in enumerate(vals, 1):
            cell = ws.cell(n, c, v)
            cell.alignment = Alignment(wrap_text=c == 5, vertical="top")
        ws.cell(n, LABEL_START + 5, reviewer)
    last = len(items) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(yn)
    dv.add(f"W2:W{last}")
    yn.add(f"X2:X{last}")
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = f"A1:AC{last}"

    g = wb.create_sheet("기준")
    g.column_dimensions["A"].width = 20
    g.column_dimensions["B"].width = 100
    rows = [
        ("라벨", "판정 기준"),
        ("NORMAL", "원문과 관련 데이터를 충분히 검토했고 정의된 의심 신호가 명확하게 없음. 사진·긴 본문만으로 자동 확정하지 않는다."),
        ("SUSPICIOUS", "텍스트·작성 행동·리뷰 간 관계에서 구체적 위험 근거를 확인함. 신호 하나로 판정하면 근거와 반대 가능성을 메모에 남긴다."),
        ("DISCLOSED_PROMO", "체험단·협찬·광고·제품 제공을 명시적으로 공개함. 의심 신호가 함께 있어도 최종 라벨은 DISCLOSED_PROMO로 하고, 의심 신호는 신호 코드에 같이 적는다(2026-11-21 결정)."),
        ("UNCERTAIN", "정보 부족, 상충하는 근거, 정상적 대안 설명이 남아 있음. 근거 부족을 정상·의심의 증거로 쓰지 않는다."),
        ("INVALID", "리뷰 본문이 아니거나 손상·오입력으로 검수 불가."),
        ("", ""),
        ("신호 코드", "설명 (여러 개면 세미콜론으로 구분, 예: TEXT_PROMOTIONAL;CROSS_PRODUCT_EXACT_DUPLICATE)"),
        *SIGNALS,
        ("", ""),
        ("정보 표시", "설명"),
        ("사진 미확인", "플랫폼이 사진 정보를 주지 않았거나 확인하지 못함. 0장과 다르다."),
        ("별점 플랫폼 미제공", "컬리는 별점을 수집하지 않는다. 5점으로 간주하지 않는다."),
        ("작성자 해시", "닉네임을 복원할 수 없게 해시한 값. masked-로 시작하면 플랫폼이 이미 가린 닉네임이라 서로 다른 사람이 같은 값일 수 있다."),
        ("다른 상품 동일 문장", "공백·기호를 뺀 본문이 같은 다른 리뷰 수 / 상품 수 (의미 문자 30자 이상만). 같은 리뷰가 여러 옵션 상품에 공유된 경우는 제외."),
        ("같은 상품·같은 날", "그 날 그 상품에 달린 리뷰 수. 상품 하루 평균과 비교해 몰림을 본다."),
        ("같은 상품·같은 시간", "같은 시(hour)에 그 상품에 달린 리뷰 수. 시각이 있는 컬리·무신사·네이버만 있다."),
        ("짧은 리뷰", "짧은 만족 표현만 있고 다른 신호가 없으면 NORMAL로 한다(2026-11-21 결정). UNCERTAIN은 신호가 있지만 정상 설명도 가능할 때만 쓴다."),
        ("", ""),
        ("절차", "두 검수자가 서로 보지 않고 같은 항목을 독립 판정한다. 모델 점수나 후보 추출 사유는 패킷에 넣지 않았다."),
    ]
    for i, (a, b) in enumerate(rows, 1):
        g.cell(i, 1, a).font = Font(bold=a in ("라벨", "신호 코드", "정보 표시", "절차"))
        g.cell(i, 2, b).alignment = Alignment(wrap_text=True)
    wb.save(path)


build(OUT / f"{NAME}.xlsx", "")

with (OUT / (f"{NAME}_key.csv" if SELECTION else "v4_pilot_key.csv")).open("w", newline="") as fh:
    keep = ["platform", "product_id", "review_id", "stratum", "author_hash"] + [f"s_{s}" for s in STRATA]
    w = csv.writer(fh)
    w.writerow(["item_id", *keep, "content_sha1"])
    for item_id, r, _ in sheet_rows():
        w.writerow([item_id, *[r[k] for k in keep], hashlib.sha1(r["content"].encode()).hexdigest()])

from collections import Counter

print("items", len(items))
print("platform", dict(Counter(r["platform"] for r in items)))
print("stratum", dict(Counter(r["stratum"] for r in items)))
