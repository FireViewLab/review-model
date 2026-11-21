# review-model

Re:view 리뷰 신뢰도의 **텍스트 신호(P_text) 모델 v4.0**을 만드는 도구다. 실제 리뷰에서 검수 후보를 뽑고, 라벨을 붙이고, KoELECTRA를 파인튜닝하고, v3.1과 비교 평가한다.

- 리뷰 본문, 라벨, 패킷, 모델 파일은 저장소에 올리지 않는다(`.gitignore`).
- 운영 서버(AI 서버, Data 서버)의 코드는 이 저장소에 없다.

## 구조

```
sql/                          운영 DB 조회 (Postgres)
  survey_*.sql                필드 보유율·하루 몰림·중복 분포 조사
  candidate_rows.sql          플랫폼별 리뷰 + 상품·작성자·중복 집계 행
src/review_model/
  constants.py                라벨·신호 코드·플랫폼 정의
  candidates/flags.py         신호 판정 → 층별 후보 풀 (작성자는 HMAC 해시)
  packet/build.py             검수 패킷(xlsx) + 키(csv)
  packet/subset.py            특정 라벨만 골라 교차 확인용 패킷 만들기
  packet/verify.py            시험 세트 확인용 패킷
  labeling/llm.py             LLM 판정 (Anthropic Messages 호환)
  labeling/consensus.py       주·보조 판정 합의
  labeling/agreement.py       일치율·κ·불일치 시트
  dataset/collect.py          라벨 모으기
  dataset/split.py            누출 없는 train/val/test 분할
  training/train.py           파인튜닝 + temperature 보정 + 임계값
  training/evaluate.py        시험 세트·프로브 평가, v3.1 비교
  training/score.py           일괄 채점 (오프라인 그림자 운영)
  policy/rules.py             등급 규칙 후보 (A 텍스트 중심, B 증거 보정, C 학습된 결합)
  policy/compare.py           후보와 v3.1을 같은 라벨·운영 표본으로 비교
  policy/config.py            채택 규칙 설정(policy.json) 내보내기·읽기
  serving/predictor.py        운영 추론: 확률 → 텍스트 점수·정책 지수·등급
  serving/app.py              HTTP 서비스 (POST /v4/analyze, GET /health)
tests/                        단위 테스트 (torch 없이 실행)
```

## 설치

```bash
uv sync --group dev                 # 데이터 준비·라벨링·테스트
uv pip install -e ".[train]" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple   # 학습·평가·채점 (GPU)
```

## 파이프라인

| 단계 | 명령 |
|---|---|
| 1. 후보 추출 | `psql ... < sql/candidate_rows.sql \| python -m review_model.candidates.flags <platform> pool` |
| 2. 검수 패킷 | `python -m review_model.packet.build --pool pool --products products.csv --context ctx.csv --out packet --name v4_main_5000 --prefix M --selection main_selection.csv` |
| 3. LLM 판정 | `python -m review_model.labeling.llm packet/v4_main_5000.xlsx llm/main_opus.xlsx claude-opus-4-8` |
| 4. 교차 확인 | `python -m review_model.packet.subset packet/v4_main_5000.xlsx llm/main_opus.xlsx SUSPICIOUS packet/main_susp.xlsx` 후 보조 모델로 3단계 |
| 5. 합의 | `python -m review_model.labeling.consensus llm/main_final.xlsx llm/main_opus.xlsx llm/main_susp_sonnet.xlsx` |
| 6. 라벨 모으기 | `python -m review_model.dataset.collect labels.csv llm/main_final.xlsx:packet/v4_main_5000_key.csv --pool pool` |
| 7. 분할 | `python -m review_model.dataset.split labels.csv split` |
| 8. 학습 | `python -m review_model.training.train split models/ptext-koelectra-v4.0` |
| 9. 평가 | `python -m review_model.training.evaluate models/ptext-koelectra-v4.0 split/test.csv eval.json --baseline <v3.1>` |
| 10. 채점 | `python -m review_model.training.score models/ptext-koelectra-v4.0 sample.csv scores.csv` |
| 11. 등급 규칙 비교 | `python -m review_model.policy.compare --val ... --test ... --out policy_compare.json` (`--help` 참고) |

모든 명령은 `--help`로 인자를 확인할 수 있다.

## 운영 추론 (등급 규칙 A: 텍스트 중심)

```bash
python -m review_model.policy.config policy_compare.json models/ptext-koelectra-v4.0/policy.json
uv pip install -e ".[train,serve]"
REVIEW_MODEL_DIR=models/ptext-koelectra-v4.0 REVIEW_MODEL_TOKEN=... \\
  uvicorn review_model.serving.app:app --host 127.0.0.1 --port 8100
```

- `POST /v4/analyze`
  - 헤더 `X-Internal-Token`이 필요하다.
  - 본문 `{platform, product_id, reviews: [{review_id, content}]}`, 최대 500건
  - 응답은 리뷰별 `text_probability`, `text_score`, `policy_index`, `level`이다.
- 등급은 의심 확률이 `warn_cut` 이상이면 WARN, `danger_cut` 이상이면 DANGER다.
- 경계값은 검증 세트에서 정밀도 0.5·0.8 기준으로 정했다.
- 사진 유무는 등급에 쓰지 않는다.

## 라벨 기준

기준과 절차 요약은 [docs/labeling_guide.md](docs/labeling_guide.md)에 있다.

| 라벨 | 뜻 | 학습 |
|---|---|---|
| NORMAL | 정의된 의심 신호가 없음 | 0 |
| SUSPICIOUS | 텍스트·행동·관계에서 구체적 위험 근거가 있음 | 1 |
| DISCLOSED_PROMO | 체험단·협찬·제공을 밝힘(의심 신호가 같이 있어도 이 라벨) | 제외 |
| UNCERTAIN | 근거가 부족하거나 상충함 | 제외 |
| INVALID | 리뷰 본문이 아님 | 제외 |

- 짧은 만족 표현만 있으면 NORMAL이다.
- 칭찬이나 홍보 문구만으로는 SUSPICIOUS가 아니다.
- 신호 코드는 `constants.SIGNALS`, 판정 지침은 `labeling/prompt.py`에 있다.

## 학습 기본값

| 항목 | 값 |
|---|---|
| 기본 모델 | `monologg/koelectra-base-v3-discriminator` |
| 학습 설정 | 4 epoch, batch 32, lr 3e-5, warmup 10% 후 선형 감소, 최대 256토큰, bf16 |
| 손실 | 클래스 가중 교차 엔트로피 |
| 모델 선택 | val PR-AUC가 가장 좋은 epoch |
| 보정 | val로 맞춘 temperature를 실제 점수 계산에 쓴다 |
| 임계값 | val에서 정밀도 0.8 이상인 지점 중 재현율이 가장 큰 곳 |

## 한계

- v4.0 라벨은 LLM 두 개가 판정하고 합의한 것이다. 사람 2인 독립 검수가 아니다.
- 다른 계열 모델(GPT)로 시험 세트 150건을 교차 확인했고, 의심 여부 κ는 0.64였다.
- 이 저장소는 텍스트 신호만 다룬다. 최종 등급을 만드는 행동·네트워크 신호와 등급 규칙은 검증되지 않았다.
