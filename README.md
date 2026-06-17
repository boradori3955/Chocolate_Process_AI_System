# 🏭 초콜릿 공정 품질 분석 AI 시스템

공정 센서 데이터(CSV)를 기반으로 불량을 예측하고, 결과를 PDF 보고서로 출력하는 **Streamlit 웹 애플리케이션**입니다.

---

## 📋 주요 기능

| 탭 | 설명 |
|---|---|
| 🧠 **공정 데이터 모델 생성** | 센서 CSV를 업로드해 불량 분류 ML 모델을 학습하고 `.pkl`로 저장 |
| 🔮 **공정 데이터 모델로 불량 예측** | 학습된 모델(또는 업로드 모델)로 새 센서 데이터의 불량 유형 예측 |
| 📊 **공정 품질 분석 보고서** | 예측 결과 · 공정별 센서 이상 분석을 PDF 보고서로 자동 생성 |

---

## 🍫 분석 대상 공정 및 센서

6개 공정의 8종 센서 데이터를 분석합니다.

| 공정 | 주요 센서 |
|---|---|
| 로스팅 | 온도, 압력, 연기 ADC |
| 분쇄 | 온도, 진동, 입자크기 |
| 콘칭 | 온도, 점도, 습도 |
| 템퍼링 | 온도, 점도 |
| 몰딩 | 온도, 압력, 습도 |
| 냉각 | 온도, 냉각온도, 습도 |

**불량 클래스**

- `Bloom` — 블루밍 (표면 흰색·회색 얼룩)
- `Crack` — 균열 (표면·내부 균열)
- `Crack_Bloom` — 균열 + 블루밍 복합
- `NoDefects` — 정상

---

## 🤖 지원 ML 모델

| 모델 | 패키지 |
|---|---|
| Random Forest | scikit-learn |
| SVM | scikit-learn |
| Logistic Regression | scikit-learn |
| MLP (신경망) | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |

- 클래스 불균형 처리: **SMOTE 오버샘플링** 또는 **클래스 가중치 자동 조정**
- 결측치 처리 전략: 공정별 중앙값 / 평균값 / 최빈값 채우기 또는 행 제거 선택 가능
- 학습된 모델은 `.pkl` 파일로 다운로드해 예측 탭에서 재사용 가능

---

## 📂 디렉토리 구조

```
chocolate-process/
├── chocolate_process.py   # 메인 앱
├── sensor_data.csv        # 학습·예측용 센서 데이터 (선택)
└── requirements.txt
```

---

## 🛠️ 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
joblib
reportlab
```

> 패키지가 일부 없어도 앱은 실행됩니다. 누락된 패키지는 앱 내 설치 안내 메시지가 표시됩니다.

### 2. 앱 실행

```bash
streamlit run chocolate_process.py
```

---

## 📊 입력 데이터 형식 (CSV)

### 학습 / 예측용 센서 CSV

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `product_id` | ✅ | 제품 고유 ID |
| `defect_label` | ✅ (학습 시) | 불량 유형 (`Bloom` / `Crack` / `Crack_Bloom` / `NoDefects`) |
| `defect_class` | — | 불량 클래스 번호 (선택) |
| `process` | ✅ | 공정명 (`로스팅` / `분쇄` / `콘칭` / `템퍼링` / `몰딩` / `냉각`) |
| `timestamp` | — | 측정 시각 |
| `temperature_c` | — | 온도 (°C) |
| `humidity_pct` | — | 습도 (%RH) |
| `pressure_bar` | — | 압력 (bar) |
| `viscosity_cp` | — | 점도 (cP) |
| `vibration_hz` | — | 진동 (Hz) |
| `smoke_adc` | — | 연기 ADC |
| `cooling_temp_c` | — | 냉각온도 (°C) |
| `particle_size_um` | — | 입자크기 (μm) |

> 한 제품(`product_id`)이 여러 공정 행을 가집니다. 앱이 내부적으로 제품 × 공정\_센서 와이드 포맷으로 피벗합니다.

---

## 📈 모델 학습 흐름

```
CSV 업로드
    ↓
결측치 처리 (전략 선택)
    ↓
공정별 피처 피벗 (wide format)
    ↓
SMOTE / 클래스 가중치 설정
    ↓
모델 선택 & 학습 (train/test split)
    ↓
성능 평가 (Accuracy / F1 / AUC / Confusion Matrix)
    ↓
모델 .pkl 다운로드
```

---

## 📄 PDF 보고서 구성

1. **예측 결과 요약** — 총 검사 수, 불량/정상 수량, 불량률 도넛 차트
2. **불량 유형별 공정·센서 영향 분석** — 불량별 영향 공정 순위 & Top 5 원인 센서
3. **제품별 예측 결과 목록** — 제품 ID / 예측 레이블 / 신뢰도 / 실제 레이블(있을 경우)

---

## 🖥️ 환경 요건

| 항목 | 권장 |
|---|---|
| Python | 3.9 이상 |
| OS | Windows / macOS / Linux |
| 한글 폰트 | 맑은 고딕 (Windows) / NanumGothic (Linux) / AppleSDGothicNeo (macOS) |

---

