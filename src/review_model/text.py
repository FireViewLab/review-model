"""본문 정규화. 중복 판정과 의미 문자 수 계산이 같은 규칙을 쓰도록 한 곳에 둔다."""

import re

_NON_MEANINGFUL = re.compile(r"[^0-9a-z가-힣]")


def normalize(text: str) -> str:
    """소문자로 바꾸고 한글·영숫자만 남긴다. 중복 판정 키로 쓴다."""
    return _NON_MEANINGFUL.sub("", str(text).lower())


def meaningful_length(text: str) -> int:
    """공백·문장부호·이모지를 뺀 글자 수."""
    return len(re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)))
