"""운영에 쓰는 등급 규칙 설정(policy.json).

채택한 규칙은 A(텍스트 중심)다. 경계값은 검증 세트로 정한 값을 compare 결과에서 내보낸다.

    python -m review_model.policy.config policy_compare.json models/ptext-koelectra-v4.0/policy.json
"""

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

from review_model.policy.rules import Cuts

POLICY_VERSION = "text-first-v1"


@dataclass(frozen=True)
class PolicyConfig:
    version: str
    warn_cut: float
    danger_cut: float
    warn_target_precision: float = 0.5
    danger_target_precision: float = 0.8

    @property
    def cuts(self) -> Cuts:
        return Cuts(warn=self.warn_cut, danger=self.danger_cut)

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PolicyConfig":
        data = json.loads(pathlib.Path(path).read_text())
        config = cls(**data)
        if not 0 < config.warn_cut < config.danger_cut <= 1:
            raise ValueError(f"잘못된 경계값: warn {config.warn_cut}, danger {config.danger_cut}")
        return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare_json", help="policy.compare 결과")
    ap.add_argument("out", help="policy.json 저장 경로(보통 모델 디렉터리 안)")
    a = ap.parse_args(argv)
    cuts = json.loads(pathlib.Path(a.compare_json).read_text())["cuts"]["A_B"]
    config = PolicyConfig(version=POLICY_VERSION, warn_cut=cuts["warn"], danger_cut=cuts["danger"])
    config.save(a.out)
    print(f"{config.version} warn {config.warn_cut:.4f} danger {config.danger_cut:.4f} → {a.out}")


if __name__ == "__main__":
    main()
