"""라벨·신호·플랫폼 정의. 패킷, LLM 지침, 학습이 모두 이 값을 쓴다."""

PLATFORM_NAMES = {
    "oliveyoung": "올리브영",
    "kurly": "컬리",
    "elevenst": "11번가",
    "musinsa": "무신사",
    "ohouse": "오늘의집",
    "naver": "네이버",
}

# 작성 시각이 초 단위로 남는 플랫폼. 나머지는 날짜만 있다.
SECOND_PRECISION_PLATFORMS = frozenset({"kurly", "musinsa", "naver"})

# 빈 사진 목록을 "0장"으로 확인할 수 있는 플랫폼. 나머지는 "미확인"이다.
EMPTY_IMAGES_VERIFIED = frozenset({"kurly"})

LABELS = ("NORMAL", "SUSPICIOUS", "DISCLOSED_PROMO", "UNCERTAIN", "INVALID")

# 텍스트 이진 모델의 학습 대상. 나머지 라벨은 학습에서 뺀다.
TARGET = {"NORMAL": 0, "SUSPICIOUS": 1}

# 신호 코드(팀 시범 검수와 같은 이름). 여러 개는 세미콜론으로 구분한다.
SIGNALS = (
    ("TEXT_PRODUCT_DESCRIPTION", "상품 설명문 형태"),
    ("TEXT_UNTRIED_CLAIM", "미사용 상태의 효과 단정"),
    ("TEXT_PROMOTIONAL", "홍보성·판매 유도"),
    ("TEXT_OVERCONFIDENT", "과도한 효과 확신"),
    ("TEXT_REPEAT", "리뷰 내부 비정상적 반복"),
    ("LOW_INFO_COMBINED", "사진 0장(확인)·5점·의미 문자 10자 미만"),
    ("DATE_CONCENTRATION", "같은 상품 하루 몰림"),
    ("HOUR_CONCENTRATION", "같은 상품 한 시간 몰림 (컬리·무신사·네이버)"),
    ("RATING_SKEW", "상품 별점 분포가 플랫폼 평소와 크게 다름"),
    ("HELPFUL_ANOMALY", "도움돼요 수 특이"),
    ("CROSS_PRODUCT_EXACT_DUPLICATE", "다른 상품에 같은 문장"),
    ("SAME_PRODUCT_EXACT_DUPLICATE", "같은 상품에 같은 문장"),
    ("AUTHOR_REPEAT", "같은 작성자 반복 작성"),
    ("DISCLOSED_PROMO", "광고·협찬·제공 표시"),
)
SIGNAL_CODES = tuple(code for code, _ in SIGNALS)

# 후보 추출 층. random은 무작위 대조군이다.
STRATA = ("combo", "promo", "unused", "certain", "repeat", "spec", "disclosed", "burst", "network", "author")

# 검수 결과 표의 ID 컬럼 후보(우리 패킷, 팀 시범 검수, 시험 세트 확인용).
ID_COLUMNS = ("항목ID", "annotation_id", "확인ID")
