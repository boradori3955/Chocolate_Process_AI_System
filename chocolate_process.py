# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, io, os, zipfile, shutil, tempfile, json, random, re, joblib
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

# ── 선택적 패키지 임포트 & 누락 감지
def _try_import(pkg_import, pip_name):
    """임포트 시도 후 실패 시 (False, pip_name) 반환"""
    try:
        __import__(pkg_import)
        return True, None
    except ImportError:
        return False, pip_name

# ── 필수 패키지 목록 (import명, pip명)
_REQUIRED_PKGS = [
    ("sklearn",        "scikit-learn"),
    ("imblearn",       "imbalanced-learn"),
    ("xgboost",        "xgboost"),
    ("lightgbm",       "lightgbm"),
    ("catboost",       "catboost"),
    ("joblib",         "joblib"),
]

_missing_pkgs = []
for _imp, _pip in _REQUIRED_PKGS:
    _ok, _pip_name = _try_import(_imp, _pip)
    if not _ok:
        _missing_pkgs.append((_imp, _pip_name))

# 누락 패키지 목록 저장 (탭 진입 시 안내 표시용)
_MISSING_PKG_NAMES = [_p for _, _p in _missing_pkgs]

# 패키지별 실제 임포트
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, roc_curve,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from lightgbm import LGBMClassifier
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

try:
    from catboost import CatBoostClassifier
    _CAT_OK = True
except ImportError:
    _CAT_OK = False

st.set_page_config(page_title="🏭 제조 품질 분석 AI 시스템 — 공정 데이터", page_icon="🏭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans KR', sans-serif; }
.title-block { background:linear-gradient(135deg,#1a0a00 0%,#2d1200 100%); border-left:4px solid #d4612a; padding:24px 32px; border-radius:4px; margin-bottom:28px; }
.title-block h1 { font-family:'IBM Plex Mono',monospace; color:#f0a060; font-size:1.6rem; margin:0 0 6px 0; }
.title-block p  { color:#888; margin:0; font-size:0.85rem; }
.section-header { font-family:'IBM Plex Mono',monospace; color:#d4612a; font-size:0.73rem; letter-spacing:.15em; text-transform:uppercase; border-bottom:1px solid #2a2a2a; padding-bottom:8px; margin:24px 0 14px 0; }
.stButton > button { background:#d4612a !important; color:#fff !important; border:none !important; border-radius:4px !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.95rem !important; padding:12px 0 !important; width:100% !important; }
.stButton > button:hover { background:#b85020 !important; }
.defect-card { border-radius:6px; padding:14px 18px; margin-bottom:10px; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }
.metric-card { background:#161616; border:1px solid #2a2a2a; border-radius:6px; padding:16px 12px; text-align:center; }
.metric-card .val { font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:#f0a060; }
.metric-card .val.red { color:#e05050; }
.metric-card .lbl { color:#666; font-size:0.75rem; margin-top:4px; }
.proc-badge { display:inline-block; padding:3px 12px; border-radius:3px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:600; margin-bottom:8px; }
.log-box { background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; padding:16px; font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:#aaa; line-height:1.9; max-height:260px; overflow-y:auto; }
/* chat_input 하단 고정 */
[data-testid="stChatInput"] {
    position: fixed; bottom: 1.5rem; left: 50%;
    transform: translateX(-50%);
    width: clamp(320px, 60vw, 860px);
    z-index: 999; border-radius: 0.75rem;
    box-shadow: 0 -2px 16px rgba(0,0,0,0.15); padding: 0.2rem;
}
[data-testid="stChatMessageContainer"], .stChatMessage { padding-bottom: 5.5rem; }
[data-testid="stBottom"] { background: transparent; border-top: none; }
.auto-badge {
    display:inline-block; background:#0d1a0d; border:1px solid #4caf50;
    color:#81c784; border-radius:4px; padding:4px 10px;
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e0e0e","axes.facecolor":"#161616",
    "axes.edgecolor":"#2a2a2a","axes.labelcolor":"#888",
    "xtick.color":"#555","ytick.color":"#555",
    "text.color":"#ccc","grid.color":"#222","grid.linestyle":"--",
})

import matplotlib.font_manager as fm
_korean_fonts = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for _fp in _korean_fonts:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _fname = fm.FontProperties(fname=_fp).get_name()
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

st.markdown("""
<div class="title-block">
  <h1>🏭 제조 품질 분석 AI 시스템</h1>
  <p>공정 데이터 모델 생성  |  공정 데이터 모델로 불량 예측  |  비전 모델 생성  |  비전 모델로 불량 검사  |  AI 챗봇 질의응답  |  품질 분석 보고서</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📸 AI 학습용 이미지 촬영 가이드", expanded=False):
    st.markdown(
        '<p style="font-size:0.9rem; line-height:2.2; margin:0;">'
        '✅ &nbsp;제품 분류 작업대 위에서 촬영 — 실제 검수 환경과 동일한 배경·조명 유지<br>'
        '📐 &nbsp;분류대 중앙에 제품 놓고 위에서 수직으로, 거리 20~40cm 유지<br>'
        '💡 &nbsp;해상도 640px 이상, 그림자 최소화, 클래스당 30장 이상 수집<br>'
        '⚠️ &nbsp;사무실·창고 등 현장과 다른 곳에서 찍으면 모델 정확도가 크게 낮아집니다.'
        '</p>',
        unsafe_allow_html=True
    )

VER_COLORS    = {"yolov5nu.pt":"#f0e060","yolov8n.pt":"#60c0f0","yolo11n.pt":"#f0a060"}
DEFECT_COLORS = {"Bloom":"#f0e060","Crack":"#f06060","Crack_Bloom":"#f0a060","NoDefects":"#81c784"}

tab_train_sensor, tab_predict, tab_report_sensor = st.tabs([
    "🧠 공정 데이터 모델 생성",
    "🔮 공정 데이터 모델로 불량 예측",
    "📊 공정 품질 분석 보고서",
])


with tab_train_sensor:
    st.markdown("""
    <div style="background:#0d1520;border-left:4px solid #60c0f0;padding:18px 24px;border-radius:4px;margin-bottom:20px">
      <h3 style="color:#60c0f0;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">🧠 공정 데이터 모델 생성</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">
        공정 센서 데이터(CSV)를 업로드하면 불량 유형을 분류하는 AI 모델을 생성하고 다운로드할 수 있습니다.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 누락 패키지 설치 안내
    if _MISSING_PKG_NAMES:
        st.markdown(
            '<div style="background:#1a1200;border-left:4px solid #f0a060;border-radius:4px;'
            'padding:16px 20px;margin-bottom:16px">'
            '<p style="color:#f0a060;font-family:monospace;font-size:0.9rem;margin:0 0 8px 0">'
            '⚠️ 아래 패키지가 설치되어 있지 않습니다. 터미널에서 설치 후 앱을 재시작하세요.</p>'
            '<code style="background:#0a0a0a;color:#ccc;padding:6px 12px;border-radius:3px;'
            f'font-size:0.85rem">pip install {" ".join(_MISSING_PKG_NAMES)}</code>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

    if not _SKLEARN_OK:
        st.error("❌ scikit-learn이 없습니다. 터미널에서 `pip install scikit-learn` 을 실행하세요.")
    else:
        # ── 공정·센서 관련 공통 상수
        _T5_DEFECT_LABELS = ["Bloom", "Crack", "Crack_Bloom", "NoDefects"]
        _T5_DEFECT_COLORS = {
            "Bloom": "#f0e060", "Crack": "#f06060",
            "Crack_Bloom": "#f0a060", "NoDefects": "#81c784",
        }
        _T5_SENSOR_COLS = [
            "temperature_c", "humidity_pct", "pressure_bar", "viscosity_cp",
            "vibration_hz", "smoke_adc", "cooling_temp_c", "particle_size_um",
        ]
        _T5_PROC_ORDER = ["로스팅", "분쇄", "콩칭", "템퍼링", "몰딩", "냉각"]
        # CSV 내 공정명을 표준 공정명으로 보정
        _T5_PROC_ORDER = list(_T5_PROCESSES.keys())

        t5_uploaded = st.file_uploader(
            "학습용 CSV 파일 업로드",
            type="csv", key="t5_csv_upload",
            help="필수 컬럼: product_id, defect_label, process, 센서값 컬럼들"
        )

        if not t5_uploaded:
            st.info("👆 학습용 센서 데이터 CSV를 업로드하세요.")
        else:
            t5_df = pd.read_csv(t5_uploaded)

            # ── 필수 컬럼 존재 여부 확인
            _t5_req = {"product_id", "defect_label", "process"}
            _t5_missing_cols = _t5_req - set(t5_df.columns)
            if _t5_missing_cols:
                st.error(f"❌ 필수 컬럼 누락: {_t5_missing_cols}")
                st.stop()

            _t5_avail_sensors = [c for c in _T5_SENSOR_COLS if c in t5_df.columns]
            _t5_label_dist    = t5_df.groupby("product_id")["defect_label"].first().value_counts()

            st.markdown(
                f'<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;'
                f'padding:10px 16px;font-size:0.82rem;color:#888;margin-bottom:12px">'
                f'📊 <b style="color:#ccc">{t5_df["product_id"].nunique()}개 제품</b> · '
                f'<b style="color:#ccc">{len(_t5_avail_sensors)}개 센서</b> · '
                f'<b style="color:#ccc">{t5_df["process"].nunique()}개 공정</b>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── 불량 레이블 결측 제품 제외
            _t5_lbl_row_miss = int(t5_df["defect_label"].isna().sum())
            _t5_cls_row_miss = int(t5_df["defect_class"].isna().sum()) \
                               if "defect_class" in t5_df.columns else 0

            _t5_pid_any_miss = set()
            if _t5_lbl_row_miss > 0:
                _t5_pid_any_miss |= set(
                    t5_df.loc[t5_df["defect_label"].isna(), "product_id"].unique()
                )
            if _t5_cls_row_miss > 0:
                _t5_pid_any_miss |= set(
                    t5_df.loc[t5_df["defect_class"].isna(), "product_id"].unique()
                )
            if _t5_pid_any_miss:
                t5_df = t5_df[~t5_df["product_id"].isin(_t5_pid_any_miss)].reset_index(drop=True)

            # ── 불량 유형 레이블 분포 시각화
            st.markdown('<div class="section-header">📋 레이블 분포</div>', unsafe_allow_html=True)
            _t5_lbl_cols = st.columns(len(_t5_label_dist))
            for _col, (_lbl, _cnt) in zip(_t5_lbl_cols, _t5_label_dist.items()):
                _dc = _T5_DEFECT_COLORS.get(str(_lbl), "#888")
                with _col:
                    st.markdown(
                        f'<div class="metric-card"><div class="val" style="color:{_dc}">{_cnt}</div>'
                        f'<div class="lbl">{_lbl}</div></div>', unsafe_allow_html=True)

            # ── 공정별 센서 데이터 시계열 시각화
            st.markdown('<div class="section-header">🏭 공정별 센서 분포</div>', unsafe_allow_html=True)
            _t5_proc_tabs = st.tabs(_T5_PROC_ORDER)
            for _ptab, _pname in zip(_t5_proc_tabs, _T5_PROC_ORDER):
                with _ptab:
                    _pinfo   = _T5_PROCESSES.get(_pname, {})
                    _pcolor  = _pinfo.get("color", "#888")
                    _sensors = _pinfo.get("sensors", _t5_avail_sensors)
                    _psub    = t5_df[t5_df["process"] == _pname].reset_index(drop=True)
                    if _psub.empty:
                        st.warning(f"{_pname} 공정 데이터 없음")
                        continue

                    st.markdown(f'<span class="proc-badge" style="background:{_pcolor}22;border:1px solid {_pcolor};color:{_pcolor}">● {_pname} ({len(_psub)}건)</span>', unsafe_allow_html=True)

                    _valid_s = [s for s in _sensors if s in _psub.columns and _psub[s].notna().any()]
                    if not _valid_s:
                        st.warning("유효한 센서 데이터 없음")
                        continue

                    # 불량 유형별 색상으로 센서 시계열 시각화
                    _fig_s, _axes_s = plt.subplots(len(_valid_s), 1,
                                                    figsize=(13, 2.5 * len(_valid_s)), sharex=True)
                    if len(_valid_s) == 1: _axes_s = [_axes_s]
                    _fig_s.tight_layout(pad=2.2)
                    _x = range(len(_psub))
                    for _ax, _sensor in zip(_axes_s, _valid_s):
                        _ax.plot(_x, _psub[_sensor], color=_pcolor, linewidth=0.8, alpha=0.9)
                        _nrm = _pinfo.get("normal", {}).get(_sensor)
                        if _nrm:
                            _ax.axhline(_nrm[1], color="#e05050", linewidth=0.9, linestyle="--", alpha=0.8)
                            if _nrm[0] > 0:
                                _ax.axhline(_nrm[0], color="#e09030", linewidth=0.9, linestyle="--", alpha=0.8)
                        # 불량 유형별 배경 색
                        for _dlbl, _dcolor in _T5_DEFECT_COLORS.items():
                            if _dlbl == "NoDefects": continue
                            _mask = (_psub["defect_label"] == _dlbl).values
                            _ylim = _ax.get_ylim()
                            _ax.fill_between(_x, _ylim[0], _ylim[1], where=_mask,
                                             alpha=0.15, color=_dcolor)
                            _ax.set_ylim(_ylim)
                        _ax.set_ylabel(_T5_SENSOR_LABELS.get(_sensor, _sensor), fontsize=8)
                        _ax.grid(True, alpha=0.3)
                    _axes_s[-1].set_xlabel("샘플 인덱스", fontsize=8)
                    st.pyplot(_fig_s); plt.close()

                    # 센서별 기술통계 요약
                    _stat_df = _psub[_valid_s].describe().round(3)
                    _stat_df.columns = [_T5_SENSOR_LABELS.get(c, c) for c in _stat_df.columns]
                    st.dataframe(_stat_df, use_container_width=True)

            # ── 제품 × 공정_센서 와이드 포맷으로 피벗
            _t5_pivot_rows = []
            for _pid, _grp in t5_df.groupby("product_id"):
                _row = {"product_id": _pid, "defect_label": _grp["defect_label"].iloc[0]}
                for _pn in _T5_PROC_ORDER:
                    _psub2 = _grp[_grp["process"] == _pn]
                    for _sc in _t5_avail_sensors:
                        _val = float(_psub2[_sc].values[0]) \
                            if len(_psub2) > 0 and _sc in _psub2.columns \
                               and not pd.isna(_psub2[_sc].values[0]) \
                            else np.nan
                        _row[f"{_pn}_{_sc}"] = _val
                _t5_pivot_rows.append(_row)

            _t5_wide      = pd.DataFrame(_t5_pivot_rows)
            _t5_feat_cols = [c for c in _t5_wide.columns if c not in ("product_id", "defect_label")]
            # 피벗 후 구조적 결측값 0으로 채움
            _t5_X_raw     = _t5_wide[_t5_feat_cols].fillna(0).values

            from sklearn.preprocessing import LabelEncoder as _T5LE
            _t5_le = _T5LE()
            _t5_all_cls = sorted(set(_T5_DEFECT_LABELS) | set(_t5_wide["defect_label"].dropna().unique()))
            _t5_le.fit(_t5_all_cls)
            _t5_y_enc = _t5_le.transform(_t5_wide["defect_label"].fillna("NoDefects"))

            with st.expander("⚙️ 학습 설정", expanded=True):
                _cs1, _cs2 = st.columns(2)
                with _cs1:
                    t5_test_size = st.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05, key="t5_test_size")
                    t5_use_smote = st.checkbox("SMOTE 적용 (클래스 불균형 보정)", value=True, key="t5_smote")
                    t5_use_scale = st.checkbox("피처 스케일링", value=True, key="t5_scale")
                with _cs2:
                    st.markdown("**결측치 처리**")
                    _t5_miss_strategy = st.selectbox(
                        "처리 방법",
                        [
                            "0으로 채우기",
                            "공정별 중앙값으로 채우기",
                            "공정별 평균값으로 채우기",
                            "공정별 최빈값으로 채우기",
                            "결측치 행 제거",
                        ],
                        index=0,
                        key="t5_missing_strategy",
                        help=(
                            "• 0으로 채우기: 공정별로 쓰이지 않는 센서(구조적 결측)에 적합\n"
                            "• 공정별 중앙값: 같은 공정 내 중앙값으로 채움 (이상치에 강함)\n"
                            "• 공정별 평균값: 같은 공정 내 평균값으로 채움\n"
                            "• 공정별 최빈값: 같은 공정 내 가장 자주 등장하는 값으로 채움\n"
                            "• 결측치 행 제거: 해당 공정 핵심 센서가 없는 행 삭제"
                        )
                    )
                    st.markdown("**비교할 모델**")
                    _t5_all_model_names = list(_build_t5_model_registry(None).keys())
                    t5_selected_models  = [n for n in _t5_all_model_names
                                           if st.checkbox(n, value=True, key=f"t5_m_{n}")]

            # ── 결측치 처리 적용
            _t5_num_cols   = t5_df.select_dtypes(include=[np.number]).columns.tolist()
            _t5_total_miss = int(t5_df[_t5_num_cols].isna().sum().sum())

            t5_df = t5_df.copy()
            if _t5_total_miss > 0:
                if _t5_miss_strategy == "0으로 채우기":
                    t5_df[_t5_num_cols] = t5_df[_t5_num_cols].fillna(0)

                elif _t5_miss_strategy == "공정별 중앙값으로 채우기":
                    for _pn in t5_df["process"].unique():
                        _idx = t5_df["process"] == _pn
                        t5_df.loc[_idx, _t5_num_cols] = \
                            t5_df.loc[_idx, _t5_num_cols].fillna(
                                t5_df.loc[_idx, _t5_num_cols].median()
                            )

                elif _t5_miss_strategy == "공정별 평균값으로 채우기":
                    for _pn in t5_df["process"].unique():
                        _idx = t5_df["process"] == _pn
                        t5_df.loc[_idx, _t5_num_cols] = \
                            t5_df.loc[_idx, _t5_num_cols].fillna(
                                t5_df.loc[_idx, _t5_num_cols].mean()
                            )

                elif _t5_miss_strategy == "공정별 최빈값으로 채우기":
                    for _pn in t5_df["process"].unique():
                        _idx = t5_df["process"] == _pn
                        _mode = t5_df.loc[_idx, _t5_num_cols].mode()
                        if not _mode.empty:
                            t5_df.loc[_idx, _t5_num_cols] = \
                                t5_df.loc[_idx, _t5_num_cols].fillna(_mode.iloc[0])

                elif _t5_miss_strategy == "결측치 행 제거":
                    _keep_rows = []
                    for _pn, _psub_r in t5_df.groupby("process"):
                        _used_s  = _T5_PROCESSES.get(_pn, {}).get("sensors", _t5_avail_sensors)
                        _valid_c = [c for c in _used_s if c in _psub_r.columns]
                        if _valid_c:
                            _psub_r = _psub_r.dropna(subset=_valid_c)
                        _keep_rows.append(_psub_r)
                    t5_df = pd.concat(_keep_rows).reset_index(drop=True)

            # ── 공정 선택에 따라 피처·데이터 범위 결정
            t5_selected_proc = st.session_state.get("t5_proc_select", "전체 공정")
            if t5_selected_proc == "전체 공정":
                _t5_feat_use = _t5_feat_cols
                _t5_X_use    = _t5_X_raw
                _t5_y_use    = _t5_y_enc
            else:
                # 선택한 공정의 피처만 필터링
                _t5_feat_use = [c for c in _t5_feat_cols if c.startswith(t5_selected_proc + "_")]
                if not _t5_feat_use:
                    st.warning(f"⚠️ '{t5_selected_proc}' 공정 피처가 없습니다.")
                    st.stop()
                _t5_X_use = _t5_wide[_t5_feat_use].fillna(0).values
                _t5_y_use = _t5_y_enc



            if not t5_selected_models:
                st.warning("모델을 1개 이상 선택해주세요.")
            else:
                st.markdown('<div class="section-header">🤖 모델 학습 & 비교</div>', unsafe_allow_html=True)

                t5_selected_proc = st.selectbox(
                    "학습할 공정 선택",
                    ["전체 공정"] + _T5_PROC_ORDER,
                    key="t5_proc_select"
                )

                if st.button("▶ 학습 시작", key="t5_train_btn"):
                    from sklearn.model_selection import train_test_split as _t5_tts

                    _t5_logs   = []
                    _t5_log_ph = st.empty()
                    _t5_prog   = st.progress(0)

                    def _render_t5_log(extra=""):
                        _body = "<br>".join(_t5_logs[-12:]) + (f"<br>{extra}" if extra else "")
                        _t5_log_ph.markdown(f'<div class="log-box">{_body}</div>', unsafe_allow_html=True)

                    _render_t5_log("⚙️ 데이터 준비 중...")

                    try:
                        _t5_X_tr, _t5_X_te, _t5_y_tr, _t5_y_te = _t5_tts(
                            _t5_X_use, _t5_y_use,
                            test_size=t5_test_size, random_state=42, stratify=_t5_y_use
                        )
                    except ValueError:
                        _t5_X_tr, _t5_X_te, _t5_y_tr, _t5_y_te = _t5_tts(
                            _t5_X_use, _t5_y_use, test_size=t5_test_size, random_state=42
                        )

                    _t5_logs.append(
                        f"✅ 공정: {t5_selected_proc} · 피처: {len(_t5_feat_use)}개 · "
                        f"클래스: {list(_t5_le.classes_)} · "
                        f"train={len(_t5_X_tr)} test={len(_t5_X_te)}"
                    )

                    # ── 스케일러: SMOTE 이전 원본 train 분포로 fit
                    _t5_scaler  = StandardScaler()
                    _t5_X_tr_sc = _t5_scaler.fit_transform(_t5_X_tr)
                    _t5_X_te_sc = _t5_scaler.transform(_t5_X_te)

                    if t5_use_smote and _SMOTE_OK and len(np.unique(_t5_y_tr)) > 1:
                        try:
                            # SMOTE는 스케일된 공간에서 적용
                            _t5_X_tr_sc, _t5_y_tr = SMOTE(random_state=42).fit_resample(_t5_X_tr_sc, _t5_y_tr)
                            _vc = pd.Series(_t5_y_tr).value_counts().to_dict()
                            _t5_logs.append(f"✅ SMOTE 적용 → {_vc}")
                        except Exception as _se:
                            _t5_logs.append(f"⚠️ SMOTE 실패: {_se}")
                    elif t5_use_smote and not _SMOTE_OK:
                        _t5_logs.append("⚠️ imbalanced-learn 미설치 → SMOTE 건너뜀")
                    _t5_prog.progress(15); _render_t5_log()

                    _t5_registry   = _build_t5_model_registry("balanced")

                    _t5_results = {}
                    _t5_n_m = len(t5_selected_models)
                    for _i, _name in enumerate(t5_selected_models):
                        if _name not in _t5_registry:
                            _t5_logs.append(f"⚠️ {_name}: 패키지 미설치 → 건너뜀")
                            continue
                        _render_t5_log(f"⚙️ [{_i+1}/{_t5_n_m}] {_name} 학습 중...")
                        _t0 = time.time()
                        # 모든 모델 스케일된 데이터로 학습 (예측 탭과 동일 기준)
                        _Xtr = _t5_X_tr_sc
                        _Xte = _t5_X_te_sc

                        _mdl = _t5_registry[_name]
                        _mdl.fit(_Xtr, _t5_y_tr)
                        _elapsed = time.time() - _t0

                        _y_pred  = _mdl.predict(_Xte)
                        _y_proba = _mdl.predict_proba(_Xte)

                        _acc  = accuracy_score(_t5_y_te, _y_pred)
                        _prec = precision_score(_t5_y_te, _y_pred, average="macro", zero_division=0)
                        _rec  = recall_score(_t5_y_te, _y_pred, average="macro", zero_division=0)
                        _f1   = f1_score(_t5_y_te, _y_pred, average="macro", zero_division=0)
                        try:
                            _auc = roc_auc_score(_t5_y_te, _y_proba, multi_class="ovr", average="macro")
                        except Exception:
                            _auc = 0.0

                        _t5_results[_name] = {
                            "model":       _mdl,
                            "accuracy":    _acc,
                            "precision":   _prec,
                            "recall":      _rec,
                            "f1":          _f1,
                            "auc":         _auc,
                            "cm":          confusion_matrix(_t5_y_te, _y_pred),
                            "cm_labels":   list(_t5_le.classes_),
                            "elapsed":     _elapsed,
                            "importances": getattr(_mdl, "feature_importances_", None),
                        }
                        _t5_logs.append(
                            f"✅ {_name}  acc={_acc:.4f}  macro-F1={_f1:.4f}  ({_elapsed:.1f}s)"
                        )
                        _t5_prog.progress(15 + int((_i + 1) / _t5_n_m * 85))
                        _render_t5_log()

                    _t5_logs.append("<span style='color:#f0a060'>🎉 완료!</span>")
                    _render_t5_log()
                    st.success(f"{len(_t5_results)}개 모델 학습 완료!")
                    st.session_state["t5_results"]    = _t5_results
                    st.session_state["t5_scaler"]     = _t5_scaler
                    st.session_state["t5_FEATURES"]   = _t5_feat_cols
                    st.session_state["t5_le"]         = _t5_le
                    _best_name = max(_t5_results, key=lambda x: _t5_results[x]["f1"])
                    st.session_state["t6_trained_model"] = {
                        "clf":          _t5_results[_best_name]["model"],
                        "scaler":       _t5_scaler,
                        "le":           _t5_le,
                        "feature_cols": _t5_feat_cols,
                        "model_name":   _best_name,
                    }
                    st.info(f"💡 예측에 사용될 모델: **{_best_name}** (macro-F1 기준 최고 성능)")

            # ── 모델 학습 결과 표시
            if "t5_results" in st.session_state:
                _r5  = st.session_state["t5_results"]
                _F5  = st.session_state["t5_FEATURES"]
                _le5 = st.session_state["t5_le"]

                # 모델별 성능 비교 테이블
                st.markdown('<div class="section-header">📊 모델 성능 비교 (macro avg)</div>', unsafe_allow_html=True)
                _mdf5 = pd.DataFrame({
                    _n: {
                        "Accuracy":    round(_v["accuracy"],  4),
                        "Precision":   round(_v["precision"], 4),
                        "Recall":      round(_v["recall"],    4),
                        "F1-Score":    round(_v["f1"],        4),
                        "AUC-ROC":     round(_v["auc"],       4),
                        "학습시간(s)": round(_v["elapsed"],   2),
                    }
                    for _n, _v in _r5.items()
                }).T
                st.dataframe(_mdf5.style.format("{:.4f}"), use_container_width=True)

                # 성능 지표 막대 차트
                st.markdown('<div class="section-header">📉 지표 시각화</div>', unsafe_allow_html=True)
                _mk5  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
                _n5   = list(_r5.keys())
                _c5   = [_T5_MODEL_COLORS.get(_n, "#f0a060") for _n in _n5]
                _fig5b, _ax5b = plt.subplots(1, 5, figsize=(16, 4)); _fig5b.tight_layout(pad=3.0)
                for _ax, _mk in zip(_ax5b, _mk5):
                    _vals = [_mdf5.loc[_n, _mk] for _n in _n5]
                    _brs  = _ax.bar(_n5, _vals, color=_c5, edgecolor="#2a2a2a", width=0.6)
                    _ax.set_title(_mk, fontsize=9, pad=6)
                    _ax.set_ylim(max(0, min(_vals) - 0.05), 1.05)
                    _ax.set_xticks(range(len(_n5)))
                    _ax.set_xticklabels(_n5, rotation=35, ha="right", fontsize=7)
                    _ax.grid(True, alpha=0.3, axis="y")
                    for _br, _v in zip(_brs, _vals):
                        _ax.text(_br.get_x() + _br.get_width()/2, _br.get_height() + 0.005,
                                 f"{_v:.3f}", ha="center", va="bottom", fontsize=6.5, color="#ccc")
                st.pyplot(_fig5b); plt.close()

                # 다중분류 혼동행렬 시각화
                st.markdown('<div class="section-header">🔲 혼동행렬 (4-class)</div>', unsafe_allow_html=True)
                _n5m = len(_r5)
                _nc5 = min(3, _n5m); _nr5 = (_n5m + _nc5 - 1) // _nc5
                _fig5c, _ax5c = plt.subplots(_nr5, _nc5, figsize=(5 * _nc5, 4.5 * _nr5))
                _ax5c = np.array(_ax5c).flatten()
                for _ax in _ax5c[_n5m:]: _ax.axis("off")
                for _ax, (_n, _v) in zip(_ax5c, _r5.items()):
                    _cm5  = _v["cm"]
                    _clbs = _v["cm_labels"]
                    _im5  = _ax.imshow(_cm5, cmap="YlOrBr")
                    _ax.set_xticks(range(len(_clbs))); _ax.set_xticklabels(_clbs, rotation=30, ha="right", fontsize=7)
                    _ax.set_yticks(range(len(_clbs))); _ax.set_yticklabels(_clbs, fontsize=7)
                    _ax.set_title(_n, fontsize=9, color=_T5_MODEL_COLORS.get(_n, "#f0a060"))
                    _ax.set_xlabel("예측", fontsize=7); _ax.set_ylabel("실제", fontsize=7)
                    for _i in range(len(_clbs)):
                        for _j in range(len(_clbs)):
                            _ax.text(_j, _i, str(_cm5[_i, _j]),
                                     ha="center", va="center", fontsize=9, fontweight="bold",
                                     color="white" if _cm5[_i, _j] > _cm5.max() * 0.5 else "#333")
                _fig5c.tight_layout(pad=1.5); st.pyplot(_fig5c); plt.close()

                # 피처 중요도 시각화
                _fi5 = {_n: _v for _n, _v in _r5.items() if _v["importances"] is not None}
                if _fi5:
                    st.markdown('<div class="section-header">🔑 피처 중요도 (Top 15)</div>', unsafe_allow_html=True)
                    _nfi5 = len(_fi5)
                    _fig5fi, _ax5fi = plt.subplots(1, _nfi5, figsize=(max(6, 4 * _nfi5), 5))
                    if _nfi5 == 1: _ax5fi = [_ax5fi]
                    for _ax, (_n, _v) in zip(_ax5fi, _fi5.items()):
                        _imp  = _v["importances"]
                        _imp_s = pd.Series(_imp, index=_F5).sort_values(ascending=False).head(15)
                        _bar_c = []
                        for _fn in _imp_s.index:
                            _mc = "#888"
                            for _pn, _pi in _T5_PROCESSES.items():
                                if _fn.startswith(_pn + "_"): _mc = _pi["color"]; break
                            _bar_c.append(_mc)
                        _ax.barh(
                            [_T5_SENSOR_LABELS.get(f.split("_", 1)[1] if "_" in f else f, f)
                             + f"\n({f.split('_',1)[0]})" if "_" in f else f
                             for f in _imp_s.index[::-1]],
                            _imp_s.values[::-1],
                            color=_bar_c[::-1], edgecolor="#2a2a2a"
                        )
                        _ax.set_title(_n, fontsize=9, color=_T5_MODEL_COLORS.get(_n, "#f0a060"))
                        _ax.set_xlabel("Importance", fontsize=8)
                        _ax.grid(True, alpha=0.3, axis="x")
                        _ax.tick_params(labelsize=7)
                    _fig5fi.tight_layout(pad=2.0); st.pyplot(_fig5fi); plt.close()

                # 모델별 성능 레이더 차트
                st.markdown('<div class="section-header">🕸️ 모델 종합 레이더</div>', unsafe_allow_html=True)
                _rm5  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
                _ang5 = np.linspace(0, 2*np.pi, len(_rm5), endpoint=False).tolist(); _ang5 += _ang5[:1]
                _av5  = [[_mdf5.loc[_n, _m] for _m in _rm5] for _n in _r5]
                _rmin5 = max(0, round(min(_v for _row in _av5 for _v in _row) - 0.1, 1))
                _rtk5  = [round(_rmin5 + _i * (1.0 - _rmin5) / 4, 2) for _i in range(5)]
                _fig5rad, _ax5rad = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                _ax5rad.set_facecolor("#161616")
                _ax5rad.set_theta_offset(np.pi/2); _ax5rad.set_theta_direction(-1)
                _ax5rad.set_xticks(_ang5[:-1]); _ax5rad.set_xticklabels(_rm5, fontsize=9)
                _ax5rad.set_ylim(_rmin5, 1.0); _ax5rad.set_yticks(_rtk5)
                _ax5rad.set_yticklabels([str(_t) for _t in _rtk5], fontsize=7, color="#555")
                _ax5rad.grid(color="#2a2a2a", linewidth=0.8)
                for _n in _r5:
                    _vr = [_mdf5.loc[_n, _m] for _m in _rm5]; _vr += _vr[:1]
                    _ax5rad.plot(_ang5, _vr, color=_T5_MODEL_COLORS.get(_n, "#f0a060"), linewidth=1.8, label=_n)
                    _ax5rad.fill(_ang5, _vr, color=_T5_MODEL_COLORS.get(_n, "#f0a060"), alpha=0.07)
                _ax5rad.legend(fontsize=8, facecolor="#161616", edgecolor="#333", labelcolor="#ccc",
                               loc="upper right", bbox_to_anchor=(1.35, 1.1))
                _fig5rad.patch.set_facecolor("#0e0e0e"); _fig5rad.tight_layout()
                st.pyplot(_fig5rad); plt.close()

                # 최적 모델 저장 및 다운로드
                st.markdown('<div class="section-header">💾 모델 저장</div>', unsafe_allow_html=True)
                st.markdown("학습된 모델을 `.pkl` 파일로 다운로드하세요.")
                _dl5_cols = st.columns(len(_r5))
                for _col, (_n, _v) in zip(_dl5_cols, _r5.items()):
                    with _col:
                        _save_obj5 = {
                            "model":       _v["model"],
                            "scaler":      st.session_state["t5_scaler"],
                            "le":          _le5,
                            "features":    _F5,
                            "model_name":  _n,
                            "metrics": {
                                "Accuracy":  round(_v["accuracy"],  4),
                                "Precision": round(_v["precision"], 4),
                                "Recall":    round(_v["recall"],    4),
                                "F1-Score":  round(_v["f1"],        4),
                                "AUC-ROC":   round(_v["auc"],       4),
                            }
                        }
                        _buf5 = io.BytesIO()
                        joblib.dump(_save_obj5, _buf5)
                        _buf5.seek(0)
                        st.download_button(
                            label=f"⬇ {_n}",
                            data=_buf5,
                            file_name=f"{_n}_4class.pkl",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=f"t5_dl_{_n}",
                        )



# ────────────────────────────────────────────────────────────
# TAB: 공정 데이터 모델로 불량 예측
# ────────────────────────────────────────────────────────────

with tab_predict:
    st.markdown("""
    <div style="background:#1a0d1a;border-left:4px solid #e070e0;padding:18px 24px;border-radius:4px;margin-bottom:20px">
      <h3 style="color:#e070e0;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">🔮 공정 데이터 모델로 불량 예측</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">
        생성된 공정 데이터 모델로 제품별 불량 유형을 예측하고, 어떤 공정·항목이 영향을 미쳤는지 분석해 드립니다.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 누락 패키지 설치 안내
    if _MISSING_PKG_NAMES:
        st.markdown(
            '<div style="background:#1a1200;border-left:4px solid #e070e0;border-radius:4px;'
            'padding:16px 20px;margin-bottom:16px">'
            '<p style="color:#e070e0;font-family:monospace;font-size:0.9rem;margin:0 0 8px 0">'
            '⚠️ 아래 패키지가 설치되어 있지 않습니다. 터미널에서 설치 후 앱을 재시작하세요.</p>'
            '<code style="background:#0a0a0a;color:#ccc;padding:6px 12px;border-radius:3px;'
            f'font-size:0.85rem">pip install {" ".join(_MISSING_PKG_NAMES)}</code>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

    if not _SKLEARN_OK:
        st.error("❌ scikit-learn이 없습니다. 터미널에서 `pip install scikit-learn` 을 실행하세요.")
    else:
        # ── 불량 유형·센서 관련 상수
        _T6_DEFECT_LABELS  = ["Bloom", "Crack", "Crack_Bloom", "NoDefects"]
        _T6_DEFECT_COLORS  = {
            "Bloom":       "#f0e060",
            "Crack":       "#f06060",
            "Crack_Bloom": "#f0a060",
            "NoDefects":   "#81c784",
        }
        _T6_SENSOR_COLS = [
            "temperature_c", "humidity_pct", "pressure_bar",
            "viscosity_cp", "vibration_hz", "smoke_adc",
            "cooling_temp_c", "particle_size_um",
        ]
        _T6_PROC_ORDER = ["로스팅", "분쇄", "콘칭", "템퍼링", "몰딩", "냉각"]

        # ── 모델 및 센서 CSV 업로드 UI (2컬럼)
        _t6_col1, _t6_col2 = st.columns(2)

        # ── ① 예측에 사용할 모델 선택
        with _t6_col1:
            st.markdown("**① 예측 모델 (.pkl)**")

            _t6_has_auto = "t6_trained_model" in st.session_state
            _t6_auto_info = ""
            if _t6_has_auto:
                _auto = st.session_state["t6_trained_model"]
                _t6_auto_info = f'{_auto.get("model_name","모델")} | 피처 {len(_auto.get("feature_cols",[]))}개'

            _t6_pkl_file = None
            if _t6_has_auto and not st.session_state.get("t6_model_manual_mode"):
                st.markdown(
                    f'<div class="auto-badge">✅ 학습 모델 자동 연결됨 — {_t6_auto_info}</div>',
                    unsafe_allow_html=True
                )
                if st.button("🔄 다른 모델 업로드", key="t6_model_reset", use_container_width=True):
                    st.session_state["t6_model_manual_mode"] = True
                    st.rerun()
                _use_auto = True
            else:
                _t6_pkl_file = st.file_uploader(".pkl 모델 파일", type=["pkl"], key="t6_pkl_upload")
                if _t6_pkl_file:
                    st.session_state.pop("t6_model_manual_mode", None)
                if st.session_state.get("t6_model_manual_mode") and _t6_has_auto:
                    if st.button("↩ 자동 연결 모델로 돌아가기", key="t6_model_auto_back", use_container_width=True):
                        st.session_state.pop("t6_model_manual_mode", None)
                        st.rerun()
                _use_auto = False

        # ── ② 예측용 센서 데이터 CSV 업로드
        with _t6_col2:
            st.markdown("**② 센서 데이터 CSV**")
            t6_csv = st.file_uploader(
                "센서 CSV",
                type=["csv"], key="t6_csv_upload",
                help="필수 컬럼: product_id, process, 센서값 컬럼들 / defect_label 있으면 정확도 비교 가능"
            )

        # ── 선택된 소스에서 모델 로드
        _t6_model_ready = False
        _t6_clf = _t6_scaler = _t6_le = _t6_feature_cols_model = None

        if _use_auto and _t6_has_auto:
            _saved = st.session_state["t6_trained_model"]
            _t6_clf              = _saved["clf"]
            _t6_scaler           = _saved["scaler"]
            _t6_le               = _saved["le"]
            _t6_feature_cols_model = _saved["feature_cols"]
            _t6_model_ready = True

        elif not _use_auto and _t6_pkl_file is not None:
            try:
                _pkl_data = joblib.load(_t6_pkl_file)
                _t6_clf              = _pkl_data["model"]
                _t6_scaler           = _pkl_data["scaler"]
                _t6_le               = _pkl_data["le"]
                _t6_feature_cols_model = _pkl_data["features"]
                st.session_state["t6_trained_model"] = {
                    "clf":          _t6_clf,
                    "scaler":       _t6_scaler,
                    "le":           _t6_le,
                    "feature_cols": _t6_feature_cols_model,
                    "model_name":   _pkl_data.get("model_name", "업로드 모델"),
                }
                _t6_model_ready = True
                st.markdown(
                    f'<div class="auto-badge">✅ .pkl 로드 완료 — {_pkl_data.get("model_name","모델")} | '
                    f'피처 {len(_t6_feature_cols_model)}개</div>',
                    unsafe_allow_html=True
                )
            except Exception as _pkl_e:
                st.error(f"❌ .pkl 로드 실패: {_pkl_e}")

        if not _t6_model_ready:
            st.warning("⚠️ 먼저 공정 데이터 모델 학습 탭에서 학습하거나 .pkl 파일을 업로드하세요.")
        else:
          if not t6_csv:
            st.info("👆 센서 데이터 CSV 파일을 업로드하세요.")
          else:
            try:
                _t6_raw = pd.read_csv(t6_csv)

                # ── 필수 컬럼 존재 여부 확인
                _t6_req = {"product_id", "process"}
                _t6_missing_cols = _t6_req - set(_t6_raw.columns)
                if _t6_missing_cols:
                    st.error(f"❌ 필수 컬럼 누락: {_t6_missing_cols}")
                    st.stop()

                _t6_has_label = "defect_label" in _t6_raw.columns
                _t6_avail_sensors = [c for c in _T6_SENSOR_COLS if c in _t6_raw.columns]

                st.markdown(
                    f'<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;'
                    f'padding:10px 16px;font-size:0.82rem;color:#888;margin-bottom:12px">'
                    f'📊 <b style="color:#ccc">{_t6_raw["product_id"].nunique()}개 제품</b> · '
                    f'<b style="color:#ccc">{len(_t6_avail_sensors)}개 센서</b> · '
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── 제품 × 공정_센서 와이드 포맷으로 피벗
                _t6_pivot_rows = []
                for _pid, _grp in _t6_raw.groupby("product_id"):
                    _row = {"product_id": _pid}
                    for _proc in _T6_PROC_ORDER:
                        _psub = _grp[_grp["process"] == _proc]
                        for _sc in _t6_avail_sensors:
                            _col_name = f"{_proc}_{_sc}"
                            _row[_col_name] = float(_psub[_sc].values[0]) if (len(_psub) > 0 and _sc in _psub.columns and not pd.isna(_psub[_sc].values[0] if len(_psub) > 0 else np.nan)) else np.nan
                    if _t6_has_label:
                        _row["defect_label"] = _grp["defect_label"].iloc[0]
                    _t6_pivot_rows.append(_row)

                _t6_wide = pd.DataFrame(_t6_pivot_rows)
                _t6_feature_cols = [c for c in _t6_wide.columns if c not in ("product_id", "defect_label")]

                # 결측값 0으로 대체
                _t6_X_wide = _t6_wide[_t6_feature_cols].fillna(0).values

                # ── 학습 시 피처 순서에 맞게 재정렬
                from sklearn.preprocessing import LabelEncoder as _T6LE
                from sklearn.ensemble import RandomForestClassifier as _T6RF
                from sklearn.preprocessing import StandardScaler as _T6SS

                _t6_X_realign = pd.DataFrame(_t6_X_wide, columns=_t6_feature_cols)
                for _fc in _t6_feature_cols_model:
                    if _fc not in _t6_X_realign.columns:
                        _t6_X_realign[_fc] = 0.0
                _t6_X_wide       = _t6_X_realign[_t6_feature_cols_model].values
                _t6_feature_cols = _t6_feature_cols_model

                # ── 예측 실행 버튼
                if st.button("▶ 예측 시작", key="t6_predict_btn"):
                    with st.spinner("예측 실행 중..."):
                        _t6_X_sc_pred   = _t6_scaler.transform(_t6_X_wide)
                        _t6_pred_enc    = _t6_clf.predict(_t6_X_sc_pred)
                        _t6_pred_proba  = _t6_clf.predict_proba(_t6_X_sc_pred)
                        _t6_pred_labels = _t6_le.inverse_transform(_t6_pred_enc)
                        st.session_state["t6_pred_cache"] = {
                            "wide":         _t6_wide,
                            "pred_labels":  _t6_pred_labels,
                            "pred_proba":   _t6_pred_proba,
                            "classes":      _t6_le.classes_,
                            "importances":  _t6_clf.feature_importances_,
                            "feature_cols": _t6_feature_cols,
                            "has_label":    _t6_has_label,
                        }

                # ── 예측 결과 표시 (session_state 캐시 사용) ─
                if "t6_pred_cache" not in st.session_state:
                    pass
                else:
                    _c = st.session_state["t6_pred_cache"]
                    _t6_wide        = _c["wide"]
                    _t6_pred_labels = _c["pred_labels"]
                    _t6_pred_proba  = _c["pred_proba"]
                    _t6_classes     = _c["classes"]
                    _t6_importances = _c["importances"]
                    _t6_feature_cols = _c["feature_cols"]
                    _t6_has_label   = _c["has_label"]
    
                    # ── 예측 결과 테이블
                    st.markdown('<div class="section-header">📋 제품별 예측 결과</div>', unsafe_allow_html=True)
    
                    _t6_result_rows = []
                    for _i, _pid_row in enumerate(_t6_wide["product_id"]):
                        _pred_lbl  = _t6_pred_labels[_i]
                        _proba_row = _t6_pred_proba[_i]
                        _conf      = float(_proba_row.max()) * 100
                        _row_data  = {
                            "제품 ID":    _pid_row,
                            "예측 레이블": _pred_lbl,
                            "신뢰도":     f"{_conf:.1f}%",
                            **{f"P({_cls})": f"{_proba_row[_ci]*100:.1f}%"
                               for _ci, _cls in enumerate(_t6_classes)},
                        }
                        if _t6_has_label:
                            _true_lbl = _t6_wide["defect_label"].iloc[_i]
                            _row_data["실제 레이블"] = _true_lbl
                            _row_data["일치"]       = "✅" if str(_pred_lbl) == str(_true_lbl) else "❌"
                        _t6_result_rows.append(_row_data)
    
                    _t6_result_df = pd.DataFrame(_t6_result_rows)
                    # 표시 컬럼 순서 정렬
                    _base_cols = ["제품 ID", "예측 레이블"]
                    if _t6_has_label:
                        _base_cols += ["실제 레이블", "일치"]
                    _base_cols += ["신뢰도"] + [f"P({c})" for c in _t6_classes]
                    _t6_result_df = _t6_result_df[[c for c in _base_cols if c in _t6_result_df.columns]]
    
                    def _t6_color_pred(val):
                        c = _T6_DEFECT_COLORS.get(str(val), "#888888")
                        return f"color: {c}; font-weight: 600"
    
                    _style_cols = ["예측 레이블"] + (["실제 레이블"] if _t6_has_label else [])
                    st.dataframe(
                        _t6_result_df.style.map(_t6_color_pred, subset=_style_cols),
                        use_container_width=True, hide_index=True
                    )
    
                    # 예측 결과 요약 카드
                    _t6_pred_dist = pd.Series(_t6_pred_labels).value_counts()
                    if _t6_has_label:
                        _t6_acc = sum(
                            str(p) == str(t)
                            for p, t in zip(_t6_pred_labels, _t6_wide["defect_label"])
                        ) / len(_t6_wide) * 100
                        _sum_cols = st.columns(2 + len(_t6_pred_dist))
                        with _sum_cols[0]:
                            st.markdown(
                                f'<div class="metric-card"><div class="val">{len(_t6_wide)}</div>'
                                f'<div class="lbl">총 제품 수</div></div>', unsafe_allow_html=True)
                        with _sum_cols[1]:
                            _acc_color = "#81c784" if _t6_acc >= 80 else "#f0a060" if _t6_acc >= 60 else "#f06060"
                            st.markdown(
                                f'<div class="metric-card"><div class="val" style="color:{_acc_color}">{_t6_acc:.1f}%</div>'
                                f'<div class="lbl">예측 정확도</div></div>', unsafe_allow_html=True)
                        for _ci, (_lbl, _cnt) in enumerate(_t6_pred_dist.items()):
                            with _sum_cols[2 + _ci]:
                                _dc = _T6_DEFECT_COLORS.get(str(_lbl), "#888")
                                st.markdown(
                                    f'<div class="metric-card"><div class="val" style="color:{_dc}">{_cnt}</div>'
                                    f'<div class="lbl">{_lbl}</div></div>', unsafe_allow_html=True)
                    else:
                        _sum_cols = st.columns(1 + len(_t6_pred_dist))
                        with _sum_cols[0]:
                            st.markdown(
                                f'<div class="metric-card"><div class="val">{len(_t6_wide)}</div>'
                                f'<div class="lbl">총 제품 수</div></div>', unsafe_allow_html=True)
                        for _ci, (_lbl, _cnt) in enumerate(_t6_pred_dist.items()):
                            with _sum_cols[1 + _ci]:
                                _dc = _T6_DEFECT_COLORS.get(str(_lbl), "#888")
                                st.markdown(
                                    f'<div class="metric-card"><div class="val" style="color:{_dc}">{_cnt}</div>'
                                    f'<div class="lbl">{_lbl}</div></div>', unsafe_allow_html=True)
    
                    # ── 불량 유형별 예측 분포 도넛 차트
                    st.markdown('<div class="section-header">📊 예측 분포</div>', unsafe_allow_html=True)
                    _t6_dist = pd.Series(_t6_pred_labels).value_counts()
                    _fig6a, _ax6a = plt.subplots(figsize=(5, 4))
                    _fig6a.patch.set_facecolor('#0e0e0e'); _ax6a.set_facecolor('#0e0e0e')
                    _pie_colors = [_T6_DEFECT_COLORS.get(str(l), "#888") for l in _t6_dist.index]
                    _wedges, _texts, _autotexts = _ax6a.pie(
                        _t6_dist.values,
                        labels=_t6_dist.index,
                        autopct="%1.0f%%",
                        colors=_pie_colors,
                        wedgeprops=dict(width=0.5),
                        startangle=90,
                        textprops={"color": "#ccc", "fontsize": 9},
                    )
                    for _at in _autotexts:
                        _at.set_fontsize(8); _at.set_color("#fff")
                    _ax6a.set_title("예측 레이블 분포", fontsize=10, color="#ccc", pad=10)
                    _fig6a.tight_layout()
                    st.pyplot(_fig6a); plt.close()
    
                    # ── 불량 유형별 공정·센서 영향 분석
                    st.markdown('<div class="section-header">🔬 불량 유형별 영향 분석</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<p style="color:#888;font-size:0.82rem;margin:-8px 0 12px 0">'
                        '예측 결과 기준으로 각 불량 유형에서 어떤 공정의 어떤 센서가 영향을 줬는지 분석합니다.'
                        '</p>',
                        unsafe_allow_html=True
                    )

                    # 분석 기준 선택 (실제/예측 레이블)
                    if _t6_has_label:
                        _t6_analysis_basis = st.radio(
                            "분석 기준",
                            ["예측 레이블 기준", "실제 레이블 기준"],
                            horizontal=True, key="t6_analysis_basis"
                        )
                        _t6_basis_labels = (
                            pd.Series(_t6_pred_labels)
                            if _t6_analysis_basis == "예측 레이블 기준"
                            else _t6_wide["defect_label"]
                        )
                    else:
                        _t6_analysis_basis = "예측 레이블 기준"
                        _t6_basis_labels = pd.Series(_t6_pred_labels)

                    # ── 분석용 피처 데이터프레임 구성
                    _t6_X_df = pd.DataFrame(_t6_X_wide, columns=_t6_feature_cols)
                    _t6_X_df["_basis_label"] = _t6_basis_labels.values

                    # 예측 결과에 존재하는 불량 클래스만 필터링
                    _t6_present_labels = [l for l in _T6_DEFECT_LABELS if l in _t6_X_df["_basis_label"].values and l != "NoDefects"]

                    _DLBL_KR = {
                        "Bloom": "블루밍", "Crack": "균열",
                        "Crack_Bloom": "균열+블루밍", "NoDefects": "정상"
                    }
                    _DLBL_DESC = {
                        "Bloom":       "표면에 흰색·회색 얼룩이 생기는 불량",
                        "Crack":       "표면 또는 내부에 균열이 생기는 불량",
                        "Crack_Bloom": "균열과 블루밍이 동시에 발생하는 복합 불량",
                        "NoDefects":   "불량 없음 — 정상 제품",
                    }

                    for _dlbl in _t6_present_labels:
                        _dc = _T6_DEFECT_COLORS.get(_dlbl, "#888")
                        _grp_mask   = _t6_X_df["_basis_label"] == _dlbl
                        _grp_n      = int(_grp_mask.sum())
                        _grp_mean   = _t6_X_df.loc[_grp_mask,  _t6_feature_cols].mean()
                        _other_mean = _t6_X_df.loc[~_grp_mask, _t6_feature_cols].mean()
                        _diff       = (_grp_mean - _other_mean).abs()
                        _imp_series = pd.Series(_t6_importances, index=_t6_feature_cols)
                        _score      = (_diff * _imp_series).sort_values(ascending=False)

                        # 공정별 영향도 집계
                        _proc_scores = {}
                        for _fn, _sv in _score.items():
                            for _pn in _T6_PROC_ORDER:
                                if _fn.startswith(_pn + "_"):
                                    _proc_scores[_pn] = _proc_scores.get(_pn, 0.0) + _sv
                                    break
                        _proc_rank = sorted(_proc_scores.items(), key=lambda x: x[1], reverse=True)

                        # ── 카드 헤더
                        _kr = _DLBL_KR.get(_dlbl, _dlbl)
                        _desc = _DLBL_DESC.get(_dlbl, "")
                        st.markdown(
                            f'<div style="background:#161616;border:1px solid #2a2a2a;'
                            f'border-top:3px solid {_dc};border-radius:8px;'
                            f'padding:18px 20px;margin-bottom:16px">'
                            # 제목 행
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
                            f'<span style="color:{_dc};font-size:1.1rem;font-weight:700">{_kr}</span>'
                            f'<span style="background:{_dc}22;color:{_dc};border-radius:20px;'
                            f'padding:2px 10px;font-size:0.75rem">{_dlbl}</span>'
                            f'<span style="color:#555;font-size:0.8rem;margin-left:auto">'
                            f'해당 제품 {_grp_n}개</span>'
                            f'</div>'
                            # 불량 설명
                            f'<div style="color:#666;font-size:0.8rem;margin-bottom:14px">{_desc}</div>',
                            unsafe_allow_html=True
                        )

                        # ── 주요 원인 공정 (가로 바)
                        _max_score = max(v for _, v in _proc_rank) if _proc_rank else 1
                        _proc_html = '<div style="margin-bottom:14px">'
                        _proc_html += '<div style="color:#888;font-size:0.75rem;margin-bottom:8px">📍 공정별 영향도</div>'
                        for _ri, (_pn, _pv) in enumerate(_proc_rank):
                            _pcolor = _T5_PROCESSES.get(_pn, {}).get("color", "#888")
                            _bar_w  = max(4, int(_pv / _max_score * 100))
                            _medal  = ["🥇", "🥈", "🥉"][_ri] if _ri < 3 else ""
                            _proc_html += (
                                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                                f'<span style="width:54px;color:{_pcolor};font-size:0.8rem;'
                                f'font-weight:600;text-align:right">{_pn}</span>'
                                f'<div style="flex:1;background:#222;border-radius:4px;height:10px">'
                                f'<div style="width:{_bar_w}%;background:{_pcolor};'
                                f'border-radius:4px;height:10px"></div></div>'
                                f'<span style="font-size:0.8rem">{_medal}</span>'
                                f'</div>'
                            )
                        _proc_html += '</div>'
                        st.markdown(_proc_html, unsafe_allow_html=True)

                        # ── Top 5 피처 — 직관적 표현
                        _top5 = _score.head(5)
                        _rows_html = '<div style="color:#888;font-size:0.75rem;margin-bottom:8px">🔍 주요 원인 센서 Top 5</div>'

                        for _rank_i, (_fn, _sv) in enumerate(_top5.items()):
                            for _pn in _T6_PROC_ORDER:
                                if _fn.startswith(_pn + "_"):
                                    _sn      = _fn.replace(_pn + "_", "", 1)
                                    _grp_v   = float(_grp_mean[_fn])
                                    _other_v = float(_other_mean[_fn])
                                    _diff_v  = _grp_v - _other_v
                                    _pct     = abs(_diff_v) / (_other_v + 1e-9) * 100
                                    _pcolor  = _T5_PROCESSES.get(_pn, {}).get("color", "#888")
                                    _sensor_kr = _T5_SENSOR_LABELS.get(_sn, _sn)

                                    # 방향 표현
                                    if _diff_v > 0:
                                        _arrow = "▲"
                                        _arrow_color = "#f06060"
                                        _diff_txt = f"정상 대비 +{_pct:.0f}% 높음"
                                        _unit_txt = f"(정상 {_other_v:.1f} → 불량 {_grp_v:.1f})"
                                    else:
                                        _arrow = "▼"
                                        _arrow_color = "#60a0f0"
                                        _diff_txt = f"정상 대비 {_pct:.0f}% 낮음"
                                        _unit_txt = f"(정상 {_other_v:.1f} → 불량 {_grp_v:.1f})"

                                    _rows_html += (
                                        f'<div style="display:flex;align-items:center;gap:10px;'
                                        f'padding:9px 12px;margin-bottom:6px;'
                                        f'background:#1a1a1a;border-radius:6px;'
                                        f'border-left:3px solid {_pcolor}">'
                                        # 순위
                                        f'<span style="color:#444;font-size:0.75rem;'
                                        f'width:16px;text-align:center">{_rank_i+1}</span>'
                                        # 공정 태그
                                        f'<span style="background:{_pcolor}22;color:{_pcolor};'
                                        f'border-radius:4px;padding:2px 8px;font-size:0.72rem;'
                                        f'white-space:nowrap">{_pn}</span>'
                                        # 센서명
                                        f'<span style="color:#ddd;font-size:0.88rem;'
                                        f'font-weight:600;min-width:80px">{_sensor_kr}</span>'
                                        # 변화 방향
                                        f'<span style="color:{_arrow_color};font-size:1rem;'
                                        f'font-weight:700">{_arrow}</span>'
                                        # 설명
                                        f'<span style="color:#aaa;font-size:0.82rem;flex:1">'
                                        f'{_diff_txt}</span>'
                                        # 수치
                                        f'<span style="color:#555;font-size:0.75rem;'
                                        f'white-space:nowrap">{_unit_txt}</span>'
                                        f'</div>'
                                    )
                                    break

                        st.markdown(_rows_html + '</div>', unsafe_allow_html=True)

            except Exception as _e6:
                st.error(f"오류 발생: {_e6}")
                import traceback

with tab_report_sensor:
    st.markdown("""<div style="background:#1a0d1a;border-left:4px solid #e070e0;padding:18px 24px;border-radius:4px;margin-bottom:24px">
      <h3 style="color:#e070e0;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">📊 공정 품질 분석 보고서</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">공정 데이터 모델 예측 결과를 기반으로 불량 현황과 공정·센서 영향 분석을 PDF로 생성합니다.</p>
    </div>""", unsafe_allow_html=True)

    # ── 연동 데이터 수집
    _rps_pred = st.session_state.get("t6_pred_cache")   # 예측 결과 캐시
    _rps_model = st.session_state.get("t6_trained_model")

    # ── STEP 1 : 연동 현황
    st.markdown('<div class="section-header">① 연동된 데이터 확인</div>', unsafe_allow_html=True)

    _ps1, _ps2 = st.columns(2)
    with _ps1:
        if _rps_pred:
            _pp_labels = _rps_pred["pred_labels"]
            _pp_total  = len(_pp_labels)
            _pp_defect = sum(1 for l in _pp_labels if l != "NoDefects")
            _pp_rate   = _pp_defect / _pp_total * 100 if _pp_total else 0
            st.markdown(
                f'<div style="background:#1a0d1a;border:1px solid #e070e0;border-radius:8px;padding:16px 14px">'
                f'<div style="color:#e070e0;font-family:monospace;font-size:0.75rem;margin-bottom:8px">✅ 예측 결과 연결됨</div>'
                f'<div style="display:flex;gap:16px">'
                f'<div><div style="color:#ccc;font-family:monospace;font-size:1.4rem;font-weight:600">{_pp_total}</div><div style="color:#666;font-size:0.75rem">총 제품</div></div>'
                f'<div><div style="color:#f06060;font-family:monospace;font-size:1.4rem;font-weight:600">{_pp_defect}</div><div style="color:#666;font-size:0.75rem">불량</div></div>'
                f'<div><div style="color:#f0a060;font-family:monospace;font-size:1.4rem;font-weight:600">{_pp_rate:.1f}%</div><div style="color:#666;font-size:0.75rem">불량률</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#161616;border:1px solid #333;border-radius:8px;padding:16px 14px">'
                '<div style="color:#555;font-family:monospace;font-size:0.75rem;margin-bottom:6px">⬜ 예측 미실행</div>'
                '<div style="color:#444;font-size:0.8rem">🔮 공정 데이터 모델로 불량 예측 탭에서<br>예측을 먼저 실행하세요</div>'
                '</div>', unsafe_allow_html=True)
    with _ps2:
        if _rps_model:
            _pm_name = _rps_model.get("model_name", "모델")
            _pm_feat = len(_rps_model.get("feature_cols", []))
            st.markdown(
                f'<div style="background:#0d1520;border:1px solid #60c0f0;border-radius:8px;padding:16px 14px">'
                f'<div style="color:#60c0f0;font-family:monospace;font-size:0.75rem;margin-bottom:8px">✅ 모델 연결됨</div>'
                f'<div style="color:#ccc;font-size:0.9rem;font-weight:600">{_pm_name}</div>'
                f'<div style="color:#666;font-size:0.75rem;margin-top:4px">피처 {_pm_feat}개</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#161616;border:1px solid #333;border-radius:8px;padding:16px 14px">'
                '<div style="color:#555;font-family:monospace;font-size:0.75rem;margin-bottom:6px">⬜ 모델 없음</div>'
                '<div style="color:#444;font-size:0.8rem">🧠 공정 데이터 모델 생성 탭에서<br>모델을 먼저 학습하세요</div>'
                '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2 : 보고서 생성
    st.markdown('<div class="section-header">② PDF 보고서 생성</div>', unsafe_allow_html=True)

    _rps_disabled = (_rps_pred is None)
    if _rps_disabled:
        st.warning("⚠️ 예측 결과가 없습니다. 공정 데이터 모델로 불량 예측 탭에서 예측을 먼저 실행하세요.")

    if st.button("📋 공정 PDF 보고서 생성하기", key="t7_gen_btn",
                 use_container_width=True, disabled=_rps_disabled):
        try:
            import io as _io7
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib import colors as _rc7
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer,
                Table, TableStyle, HRFlowable, Image as RLImage7
            )
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _rp7
            import matplotlib.font_manager as _rfm7

            # 한글 폰트
            _rfn7 = "Helvetica"
            for _fp7 in ["C:/Windows/Fonts/malgun.ttf",
                         "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                         "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
                if os.path.exists(_fp7):
                    try:
                        pdfmetrics.registerFont(TTFont("KR7", _fp7))
                        _rfn7 = "KR7"
                        _rfm7.fontManager.addfont(_fp7)
                        _rp7.rcParams["font.family"] = _rfm7.FontProperties(fname=_fp7).get_name()
                        _rp7.rcParams["axes.unicode_minus"] = False
                    except Exception:
                        pass
                    break

            _rb7 = _io7.BytesIO()
            _W7 = A4[0] - 4.4*cm
            _doc7 = SimpleDocTemplate(_rb7, pagesize=A4,
                                      leftMargin=2.2*cm, rightMargin=2.2*cm,
                                      topMargin=2.2*cm, bottomMargin=2.2*cm)

            def _s7(name, size=10, leading=14, color="#333333", bold=False, align="LEFT", sb=0, sa=4):
                return ParagraphStyle(name, fontName=_rfn7, fontSize=size, leading=leading,
                                      textColor=_rc7.HexColor(color),
                                      alignment={"LEFT":0,"CENTER":1,"RIGHT":2}[align],
                                      spaceBefore=sb, spaceAfter=sa, wordWrap="CJK")

            S7_TITLE = _s7("T7T", size=20, color="#1a001a", sa=4)
            S7_SUB   = _s7("T7S", size=8,  color="#999999", sa=2)
            S7_H1    = _s7("T7H1", size=13, color="#9c27b0", sb=18, sa=6)
            S7_H2    = _s7("T7H2", size=10, color="#333333", sb=10, sa=4)
            S7_BODY  = _s7("T7B", size=9,  color="#333333", leading=14, sa=3)
            S7_CAP   = _s7("T7C", size=7.5, color="#888888", align="CENTER", sa=2)

            _st7 = []
            _pc7 = st.session_state["t6_pred_cache"]
            _pl7 = _pc7["pred_labels"]
            _pp7 = _pc7["pred_proba"]
            _pw7 = _pc7["wide"]
            _pcls7 = _pc7["classes"]
            _pimp7 = _pc7["importances"]
            _pfc7  = _pc7["feature_cols"]
            _phl7  = _pc7["has_label"]

            # ── 표지
            _st7.append(Paragraph("초콜릿 공장  공정 품질 분석 보고서", S7_TITLE))
            _st7.append(Spacer(1, 3))
            _st7.append(Paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}", S7_SUB))
            if _rps_model:
                _st7.append(Paragraph(f"사용 모델: {_rps_model.get('model_name','모델')} | 피처 {len(_rps_model.get('feature_cols',[]))}개", S7_SUB))
            _st7.append(Spacer(1, 4))
            _st7.append(HRFlowable(width=_W7, thickness=1.5,
                                   color=_rc7.HexColor("#9c27b0"), spaceAfter=14))

            # ── 섹션 1 : 예측 결과 요약
            _st7.append(Paragraph("1. 예측 결과 요약", S7_H1))
            _pp7_total  = len(_pl7)
            _pp7_defect = sum(1 for l in _pl7 if l != "NoDefects")
            _pp7_normal = _pp7_total - _pp7_defect
            _pp7_rate   = _pp7_defect / _pp7_total * 100 if _pp7_total else 0

            _sum7 = [
                ["항목", "수량", "비율"],
                ["총 제품", str(_pp7_total), "100%"],
                ["불량 예측", str(_pp7_defect), f"{_pp7_rate:.1f}%"],
                ["정상 예측", str(_pp7_normal), f"{100-_pp7_rate:.1f}%"],
            ]
            if _phl7:
                _acc7 = sum(str(p)==str(t) for p,t in zip(_pl7, _pw7["defect_label"])) / _pp7_total * 100
                _sum7.append(["예측 정확도", f"{_acc7:.1f}%", ""])
            _t7sum = Table(_sum7, colWidths=[_W7*0.4, _W7*0.3, _W7*0.3])
            _t7sum.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  _rc7.HexColor("#9c27b0")),
                ("TEXTCOLOR",     (0,0), (-1,0),  _rc7.white),
                ("FONTNAME",      (0,0), (-1,-1), _rfn7),
                ("FONTSIZE",      (0,0), (-1,-1), 9),
                ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [_rc7.HexColor("#fafafa"), _rc7.HexColor("#f0f0f0")]),
                ("GRID",          (0,0), (-1,-1), 0.3, _rc7.HexColor("#cccccc")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            _st7.append(_t7sum)
            _st7.append(Spacer(1, 8))

            # 예측 분포 도넛
            _pd7 = {}
            for _l7 in _pl7: _pd7[_l7] = _pd7.get(_l7, 0) + 1
            _DC7 = {"Bloom":"#f0e060","Crack":"#f06060","Crack_Bloom":"#f0a060",
                    "NoDefects":"#81c784","Defect":"#e07070","Error":"#888888"}
            _fig7a, _ax7a = _rp7.subplots(figsize=(5, 3.2))
            _fig7a.patch.set_facecolor("white"); _ax7a.set_facecolor("white")
            _ax7a.pie(list(_pd7.values()), labels=list(_pd7.keys()),
                      autopct="%1.0f%%",
                      colors=[_DC7.get(l,"#888") for l in _pd7],
                      wedgeprops=dict(width=0.5), startangle=90,
                      textprops={"color":"#333","fontsize":8})
            _ax7a.set_title("예측 레이블 분포", fontsize=10, color="#333", pad=8)
            _fig7a.tight_layout()
            _buf7a = _io7.BytesIO()
            _fig7a.savefig(_buf7a, format="png", dpi=130, bbox_inches="tight", facecolor="white")
            _rp7.close(_fig7a); _buf7a.seek(0)
            _st7.append(RLImage7(_buf7a, width=_W7*0.55, height=_W7*0.36))
            _st7.append(Spacer(1, 6))

            # ── 섹션 2 : 불량 유형별 공정·센서 영향 분석
            _st7.append(Paragraph("2. 불량 유형별 공정·센서 영향 분석", S7_H1))
            _T6P  = ["로스팅","분쇄","콘칭","템퍼링","몰딩","냉각"]
            _T6SK = {"temperature_c":"온도","humidity_pct":"습도","pressure_bar":"압력",
                     "viscosity_cp":"점도","vibration_hz":"진동","smoke_adc":"연기",
                     "cooling_temp_c":"냉각온도","particle_size_um":"입자크기"}
            _DLBL_KR7 = {"Bloom":"블루밍","Crack":"균열","Crack_Bloom":"균열+블루밍"}

            _imp7s     = pd.Series(_pimp7, index=_pfc7)
            _X7_df     = pd.DataFrame(
                            np.zeros((len(_pl7), len(_pfc7))), columns=_pfc7)
            # wide DataFrame에서 피처값 복원
            for _fc7 in _pfc7:
                if _fc7 in _pw7.columns:
                    _X7_df[_fc7] = _pw7[_fc7].fillna(0).values

            _X7_df["_lbl"] = _pl7

            _defect_lbls7 = [l for l in ["Bloom","Crack","Crack_Bloom"] if l in _X7_df["_lbl"].values]

            for _dlbl7 in _defect_lbls7:
                _dc7       = {"Bloom":"#c8a000","Crack":"#c03030","Crack_Bloom":"#b06000"}.get(_dlbl7,"#888")
                _kr7       = _DLBL_KR7.get(_dlbl7, _dlbl7)
                _gmask7    = _X7_df["_lbl"] == _dlbl7
                _gn7       = int(_gmask7.sum())
                _gmean7    = _X7_df.loc[_gmask7,  _pfc7].mean()
                _omean7    = _X7_df.loc[~_gmask7, _pfc7].mean()
                _diff7     = (_gmean7 - _omean7).abs()
                _score7    = (_diff7 * _imp7s).sort_values(ascending=False)

                # 공정별 영향도
                _ps7 = {}
                for _fn7 in _score7.index:
                    for _pn7 in _T6P:
                        if _fn7.startswith(_pn7 + "_"):
                            _ps7[_pn7] = _ps7.get(_pn7, 0.0) + float(_score7[_fn7])
                            break
                _prank7 = sorted(_ps7.items(), key=lambda x: x[1], reverse=True)[:3]

                # 불량 유형 헤더
                _st7.append(Spacer(1, 6))
                _st7.append(Paragraph(
                    f"▶  {_kr7}  ({_dlbl7})  —  {_gn7}개 제품",
                    ParagraphStyle("T7DH", fontName=_rfn7, fontSize=10,
                                   textColor=_rc7.HexColor(_dc7),
                                   spaceBefore=8, spaceAfter=4, wordWrap="CJK")
                ))

                # 영향 공정 순위 표
                _rank_data = [["순위", "공정", "영향도(상대)"]]
                _max_ps7 = _prank7[0][1] if _prank7 else 1
                for _ri7, (_pn7, _pv7) in enumerate(_prank7):
                    _bar = "■" * max(1, int(_pv7 / _max_ps7 * 10))
                    _rank_data.append([f"{_ri7+1}위", _pn7, _bar])
                _rt7 = Table(_rank_data, colWidths=[_W7*0.15, _W7*0.25, _W7*0.6])
                _rt7.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0),  _rc7.HexColor("#444444")),
                    ("TEXTCOLOR",     (0,0), (-1,0),  _rc7.white),
                    ("TEXTCOLOR",     (2,1), (2,-1),  _rc7.HexColor(_dc7)),
                    ("FONTNAME",      (0,0), (-1,-1), _rfn7),
                    ("FONTSIZE",      (0,0), (-1,-1), 8),
                    ("ALIGN",         (0,0), (1,-1),  "CENTER"),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [_rc7.HexColor("#fafafa"), _rc7.HexColor("#f3f3f3")]),
                    ("GRID",          (0,0), (-1,-1), 0.3, _rc7.HexColor("#cccccc")),
                    ("TOPPADDING",    (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                _st7.append(_rt7)
                _st7.append(Spacer(1, 6))

                # Top 5 센서 영향 표
                _st7.append(Paragraph("주요 원인 센서 Top 5",
                    ParagraphStyle("T7SH", fontName=_rfn7, fontSize=8.5,
                                   textColor=_rc7.HexColor("#555555"),
                                   spaceBefore=4, spaceAfter=3)))
                _s5_data = [["공정", "센서", "변화", "정상 평균", "불량 평균"]]
                for _fn7, _ in _score7.head(5).items():
                    for _pn7 in _T6P:
                        if _fn7.startswith(_pn7 + "_"):
                            _sn7   = _fn7.replace(_pn7 + "_", "", 1)
                            _gv7   = float(_gmean7[_fn7])
                            _ov7   = float(_omean7[_fn7])
                            _arrow = "▲ 높음" if _gv7 > _ov7 else "▼ 낮음"
                            _s5_data.append([
                                _pn7,
                                _T6SK.get(_sn7, _sn7),
                                _arrow,
                                f"{_ov7:.1f}",
                                f"{_gv7:.1f}",
                            ])
                            break
                _s5t = Table(_s5_data,
                             colWidths=[_W7*0.15, _W7*0.2, _W7*0.2, _W7*0.22, _W7*0.23])
                _up_color   = _rc7.HexColor("#c03030")
                _down_color = _rc7.HexColor("#2060b0")
                _s5_style = [
                    ("BACKGROUND",    (0,0), (-1,0),  _rc7.HexColor(_dc7)),
                    ("TEXTCOLOR",     (0,0), (-1,0),  _rc7.white),
                    ("FONTNAME",      (0,0), (-1,-1), _rfn7),
                    ("FONTSIZE",      (0,0), (-1,-1), 8),
                    ("ALIGN",         (2,0), (-1,-1), "CENTER"),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [_rc7.HexColor("#fafafa"), _rc7.HexColor("#f0f0f0")]),
                    ("GRID",          (0,0), (-1,-1), 0.3, _rc7.HexColor("#cccccc")),
                    ("TOPPADDING",    (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]
                # 변화 방향 색상 적용
                for _ri7 in range(1, len(_s5_data)):
                    _clr7 = _up_color if "▲" in _s5_data[_ri7][2] else _down_color
                    _s5_style.append(("TEXTCOLOR", (2,_ri7), (2,_ri7), _clr7))
                _s5t.setStyle(TableStyle(_s5_style))
                _st7.append(_s5t)
                _st7.append(Spacer(1, 10))

            # ── 섹션 3 : 제품별 예측 결과 목록
            _st7.append(Paragraph("3. 제품별 예측 결과", S7_H1))
            _t7_list_hdr = ["제품 ID", "예측 레이블", "신뢰도"]
            if _phl7: _t7_list_hdr += ["실제 레이블", "일치"]
            _t7list_data = [_t7_list_hdr]
            for _i7, _row7 in _pw7.iterrows():
                if _i7 >= 60: break
                _plbl7 = _pl7[_i7]
                _conf7 = float(_pp7[_i7].max()) * 100
                _r7 = [str(_row7["product_id"]), str(_plbl7), f"{_conf7:.1f}%"]
                if _phl7:
                    _tlbl7 = str(_row7["defect_label"])
                    _r7 += [_tlbl7, "✅" if str(_plbl7)==_tlbl7 else "❌"]
                _t7list_data.append(_r7)
            _cw7 = [_W7*0.25, _W7*0.3, _W7*0.2]
            if _phl7: _cw7 += [_W7*0.15, _W7*0.1]
            else: _cw7[-1] = _W7 - sum(_cw7[:-1])
            _t7lst = Table(_t7list_data, colWidths=_cw7)
            _t7lst.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  _rc7.HexColor("#9c27b0")),
                ("TEXTCOLOR",     (0,0), (-1,0),  _rc7.white),
                ("FONTNAME",      (0,0), (-1,-1), _rfn7),
                ("FONTSIZE",      (0,0), (-1,0),  8),
                ("FONTSIZE",      (0,1), (-1,-1), 7.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [_rc7.HexColor("#fafafa"), _rc7.HexColor("#f0f0f0")]),
                ("GRID",          (0,0), (-1,-1), 0.3, _rc7.HexColor("#cccccc")),
                ("ALIGN",         (2,0), (-1,-1), "CENTER"),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ]))
            _st7.append(_t7lst)
            if len(_pw7) > 60:
                _st7.append(Paragraph(f"※ 전체 {len(_pw7)}건 중 상위 60건만 표시", S7_CAP))

            # ── PDF 빌드
            _doc7.build(_st7)
            _rb7.seek(0)
            st.session_state["t7_pdf_buf"] = _rb7.getvalue()
            st.success("✅ 공정 PDF 보고서 생성 완료!")

        except ImportError:
            st.error("❌ reportlab 패키지가 필요합니다: pip install reportlab")
        except Exception as _e7:
            st.error(f"❌ PDF 생성 실패: {_e7}")
            import traceback
            st.code(traceback.format_exc())

    if "t7_pdf_buf" in st.session_state:
        _fname7 = f"chocolate_process_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="⬇ 공정 PDF 보고서 다운로드",
            data=st.session_state["t7_pdf_buf"],
            file_name=_fname7,
            mime="application/pdf",
            use_container_width=True,
            key="t7_dl_btn",
        )


# ────────────────────────────────────────────────────────────
# TAB: 비전 품질 분석 보고서
# ────────────────────────────────────────────────────────────