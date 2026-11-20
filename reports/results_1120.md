# 11/18 피드백 반영 결과 (11/20)

모든 수치: 합성 사용자, 95% 부트스트랩 CI(사용자 단위). †는 짝지은 차이의 CI가 0을 포함하지 않음.

## 1. 파일럿 학습/테스트 분리 (pilot v3)

- `utils/build_pilot_v3.py` → `results/codebook/pilot_v3/questions.json`
- 학습 10문항(블록 0)을 모두 보여 준 뒤 테스트 8문항 + 반복 2문항(블록 1). 블록 안에서만 섞음.
- 페이지가 응답 순번 `n`을 기록하고, `utils/analyze_pilot.py`가 `order_ok`로 확인. 학습·테스트를 섞어 쓰던 leave-one-out(`personal_cv`)은 삭제.
- 링크: https://claude.ai/artifact/F4mKcyyLz1PBuma7geR987 (v3으로 교체), 오프라인판 `results/codebook/pilot_v3/pilot_offline.html`

## 2. 문항 수 60 → 20 (`utils/pilot_size_sim.py`, 사용자 2000명)

평가: 어느 쪽도 학습에 쓰지 않은 v2 테스트 14문항.

| 학습 문항 | 사용자 | 일치율 | 응답 예측률 | cos(θ̂, θ*) |
|---|---|---|---|---|
| v2: 42문항 | dense | 0.928 | 0.860 | 0.927 |
| **v3: 10문항 (D-optimal)** | dense | **0.876** | 0.826 | 0.816 |
| v2에서 무작위 10문항 | dense | 0.837 | 0.794 | 0.752 |
| v2: 42문항 | sparse | 0.865 | 0.788 | 0.905 |
| **v3: 10문항 (D-optimal)** | sparse | **0.792** | 0.747 | 0.724 |
| v2에서 무작위 10문항 | sparse | 0.780 | 0.734 | 0.681 |

참가자 한 명의 측정 테스트 정확도 표준편차: v2(14문항) 0.10~0.12, v3(8문항) 0.135~0.14.

## 3. DPO baseline (`src/preference/dpo.py`, `results/codebook/dpo/`)

DPO: 참조 정책 = 사람 사전확률(θ_ref = sd_human·e_human), β=0.3, 경사하강 1000스텝. β·스텝은 GC dense에서 10개 조합을 스윕해 가장 좋은 값을 고름(DPO에 유리한 조건, `results/codebook/dpo/sweep/`).

선호 regret(Simple regret, 낮을수록 좋음), 질문 5개 (20개):

| 방법 | GC dense | GC sparse | ETH dense | ETH sparse |
|---|---|---|---|---|
| Random safe candidate (선호 없음) | 0.492 | 0.493 | 0.495 | 0.490 |
| Human prior | 0.457 | 0.472 | 0.454 | 0.469 |
| BT + Random | 0.154 (0.044) | 0.178 (0.072) | 0.125 (0.042) | 0.164 (0.068) |
| BT + AQ | 0.112 (0.041) | 0.144 (0.059) | 0.109 (0.034) | 0.126 (0.053) |
| **BT + EUBO (현재)** | **0.096 (0.031)** | **0.118 (0.051)** | **0.088 (0.029)** | **0.103** (0.065) |
| DPO + Random | 0.144 (0.053) | 0.174 (0.080) | 0.116 (0.046) | 0.151 (0.081) |
| DPO + EUBO 질문 | 0.106 (0.041) | 0.117 (0.064) | 0.087 (0.032) | 0.114 (0.068) |
| 정답 θ* | 0.000 | 0.000 | 0.000 | 0.000 |

짝지은 regret 차이 (음수 = EUBO가 더 낮음), 5개 / 20개:

| | GC dense | GC sparse | ETH dense | ETH sparse |
|---|---|---|---|---|
| EUBO − DPO | −0.048† / −0.022† | −0.056† / −0.029† | −0.028† / −0.017† | −0.049† / −0.016† |
| EUBO − DPO+EUBO질문 | −0.010† / −0.010† | +0.001 / −0.013† | 0.000 / −0.003 | −0.011† / −0.003 |

- 같은 랜덤 질문에서 DPO는 BT와 비슷하고(5개 일치율 차이 +0.003~+0.010, ETH sparse만 †), 20개에서는 BT보다 약간 나쁨(일치율 −0.005~−0.016†).
- 같은 EUBO 질문을 주면 차이가 작아짐 → 이득의 대부분은 질문 선택(EUBO)에서 나오고, 추정기(BT vs DPO)는 부차적.
- Random/AQ/EUBO 수치는 기존 본 표(`paper_main_table.tex`)와 동일하게 재현됨.

## 4. 후보 생성기 × 질문 방법 (`utils/generator_comparison.py`, `results/codebook/generator/`)

같은 GC 장면 811개(Crowd-FM 후보가 이미 있는 900개 중 두 생성기 모두 안전 후보 ≥ 4), 같은 사용자 200명, 앞 3초 기준, 특징 4개(side, front, yield, detour; human 제외). 테스트 장면은 질문 장면과 다른 사람.

- 안전 후보 중앙값: Crowd-FM 20개, VQ-VAE 12개. 3초 시점 좌우 폭 중앙값: Crowd-FM 0.50 m, VQ-VAE 2.55 m.

합집합 regret(두 생성기 후보를 합친 범위로 정규화, 비교 가능한 지표), 질문 5개 (20개):

| 방법 | dense θ* | sparse θ* |
|---|---|---|
| Crowd-FM + Random | 0.397 (0.351) | 0.384 (0.344) |
| Crowd-FM + AQ | 0.368 (0.336) | 0.354 (0.316) |
| Crowd-FM + EUBO | 0.354 (0.330) | 0.351 (0.318) |
| Crowd-FM 정답 θ* (범위 한계) | 0.317 | 0.287 |
| VQ-VAE + Random | 0.157 (0.085) | 0.205 (0.127) |
| VQ-VAE + AQ | 0.123 (0.077) | 0.162 (0.101) |
| **VQ-VAE + EUBO (ours)** | **0.113 (0.076)** | **0.154** (0.108) |
| VQ-VAE 정답 θ* (범위 한계) | 0.057 | 0.068 |

짝지은 차이 (5개 / 20개):

| | dense | sparse |
|---|---|---|
| VQ+EUBO − CF+EUBO | −0.241† / −0.255† | −0.197† / −0.210† |
| CF+EUBO − CF+Random | −0.043† / −0.020† | −0.032† / −0.026† |
| VQ+EUBO − VQ+Random | −0.044† / −0.009† | −0.051† / −0.018† |
| VQ+EUBO − VQ+AQ | −0.010 / −0.002 | −0.008 / +0.008† |

- 자기 후보 안 regret(비교 불가, 참고): Crowd-FM+EUBO 0.111 (0.040), VQ+EUBO 0.066 (0.022) [dense].
- 우회를 싫어하는 사용자만(θ_detour < 0, 넓은 후보에 유리한 사용자 제외): 5개 기준 CF+EUBO 0.196 vs VQ+EUBO 0.134 (dense), 0.233 vs 0.174 (sparse). 격차는 줄지만 VQ-VAE가 여전히 낮음.
- 해석: 차이의 대부분은 후보 범위(coverage). Crowd-FM은 θ*를 알아도 0.29~0.32에서 멈춤. 질문 방법 효과(EUBO−Random)는 두 생성기에서 비슷한 크기.

## 5. regret 정식 정의

`reports/method_regret.tex` (Method 절 초안, 식 + Figure), 그림 `reports/fig_regret.pdf` (`utils/plot_regret_curves.py`).
