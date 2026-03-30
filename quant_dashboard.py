import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from strategy import run_strategy
from strategy_tuned import run_tuned_strategy
import os

st.set_page_config(page_title="PPO 강화학습 포트폴리오 최적화", page_icon="", layout="wide")

TICKERS = ["SPY", "TLT", "GLD", "DBC", "SHY"]
COLORS = {
    "basic": "#1f77b4",
    "tuned": "#2ca02c",
    "basic_dd": "#d62728",
    "tuned_dd": "#ff7f0e",
}
ASSET_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ────────────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ────────────────────────────────────────────────────────────────────
if 'basic_results' not in st.session_state:
    st.session_state.basic_results = None
if 'tuned_results' not in st.session_state:
    st.session_state.tuned_results = None
if 'auto_loaded' not in st.session_state:
    st.session_state.auto_loaded = False


# ────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ────────────────────────────────────────────────────────────────────
def compute_extra_metrics(portfolio_returns, total_return, max_drawdown):
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    return {
        "sortino": (np.mean(portfolio_returns) / downside_std) * np.sqrt(252),
        "calmar": (total_return / 100) / (max_drawdown + 1e-8),
        "win_rate": np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100,
    }


def load_model_results(model_path, strategy_type="basic"):
    from stable_baselines3 import PPO
    if strategy_type == "basic":
        from strategy import fetch_data, PortfolioEnvMonthly
    else:
        from strategy_tuned import fetch_data, PortfolioEnvMonthly

    test_df = fetch_data("2024-01-01", "2025-12-31")
    test_env = PortfolioEnvMonthly(test_df)
    model = PPO.load(model_path, env=test_env)

    obs, _ = test_env.reset()
    portfolio_values = [test_env.portfolio_value]
    weights_history = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights_history.append(action.copy())
        obs, _, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["portfolio_value"])

    final_value = portfolio_values[-1]
    total_return = (final_value - 1.0) * 100
    portfolio_returns = np.array(test_env.portfolio_returns)
    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)
    max_drawdown = max(test_env.drawdowns) if test_env.drawdowns else 0
    extra = compute_extra_metrics(portfolio_returns, total_return, max_drawdown)

    return {
        "portfolio_values": portfolio_values,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
        "returns": portfolio_returns,
        "drawdowns": test_env.drawdowns,
        "weights": np.array(weights_history),
        **extra,
    }


def auto_load_if_available():
    """앱 시작 시 저장된 모델이 있으면 자동으로 결과를 로드한다."""
    loaded_any = False
    if st.session_state.basic_results is None and os.path.exists("models/ray_dalio_portfolio_model.zip"):
        try:
            st.session_state.basic_results = load_model_results(
                "models/ray_dalio_portfolio_model", "basic"
            )
            loaded_any = True
        except Exception:
            pass
    if st.session_state.tuned_results is None and os.path.exists("models/ray_dalio_tuned_model.zip"):
        try:
            st.session_state.tuned_results = load_model_results(
                "models/ray_dalio_tuned_model", "tuned"
            )
            loaded_any = True
        except Exception:
            pass
    return loaded_any


# ────────────────────────────────────────────────────────────────────
# 자동 로드 (최초 1회)
# ────────────────────────────────────────────────────────────────────
if not st.session_state.auto_loaded:
    with st.spinner("저장된 모델 자동 로드 중..."):
        auto_load_if_available()
    st.session_state.auto_loaded = True


# ────────────────────────────────────────────────────────────────────
# 레이아웃
# ────────────────────────────────────────────────────────────────────
st.title("PPO 강화학습 포트폴리오 최적화 대시보드")
st.markdown("**기본 전략 vs Optuna 최적화 전략 성과 비교**")
st.markdown("---")

with st.sidebar:
    st.header("전략 개요")
    st.markdown("""
**투자 자산 (5개)**
- SPY: 미국 주식 (S&P 500)
- TLT: 장기 국채 (20년+)
- GLD: 금
- DBC: 원자재
- SHY: 단기 국채

**하이퍼파라미터 비교**
| 항목 | 기본 | 최적화 |
|------|------|--------|
| 활성함수 | ReLU | SiLU |
| 네트워크 | [64,64] | [64,32] |
| 엔트로피 | 0.0 | 0.0396 |
| 학습률 | 3e-4 | 2.4e-4 |
| Clip Range | 0.2 | 0.260 |
| Gamma | 0.99 | 0.995 |

**학습 기간**
- 기간: 2005–2023 (19년)
- 타임스텝: 300,000

**테스트 기간**
- 기간: 2024–2025 (2년)
- 리밸런싱: 월 1회 (20일)
""")
    st.markdown("---")
    st.caption("교육 및 연구 목적 | 투자 조언 아님")

tab1, tab2 = st.tabs(["빠른 실행 (사전학습 모델)", "직접 학습하기"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — 빠른 실행
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.header("사전학습된 모델로 결과 확인")

    basic_exists = os.path.exists("models/ray_dalio_portfolio_model.zip")
    tuned_exists = os.path.exists("models/ray_dalio_tuned_model.zip")

    # ── 버튼 영역 ──────────────────────────────────────────────────
    if basic_exists or tuned_exists:
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        with col_btn1:
            run_all = st.button(
                "두 전략 동시 실행",
                use_container_width=True,
                type="primary",
                help="기본 전략 + 최적화 전략을 한 번에 실행합니다",
            )
        with col_btn2:
            col_b, col_t = st.columns(2)
            with col_b:
                run_basic = st.button("기본만", use_container_width=True,
                                      disabled=not basic_exists)
            with col_t:
                run_tuned = st.button("최적화만", use_container_width=True,
                                      disabled=not tuned_exists)
        with col_btn3:
            if st.button("초기화", use_container_width=True):
                st.session_state.basic_results = None
                st.session_state.tuned_results = None
                st.rerun()
    else:
        st.warning("사전학습 모델 없음 — '직접 학습하기' 탭에서 먼저 학습을 실행하세요.")
        run_all = run_basic = run_tuned = False

    # ── 실행 로직 ──────────────────────────────────────────────────
    targets = []
    if 'run_all' in dir() and run_all:
        if basic_exists:
            targets.append(("basic", "models/ray_dalio_portfolio_model", "기본 전략"))
        if tuned_exists:
            targets.append(("tuned", "models/ray_dalio_tuned_model", "최적화 전략"))
    if 'run_basic' in dir() and run_basic:
        targets.append(("basic", "models/ray_dalio_portfolio_model", "기본 전략"))
    if 'run_tuned' in dir() and run_tuned:
        targets.append(("tuned", "models/ray_dalio_tuned_model", "최적화 전략"))

    for strategy_type, model_path, label in targets:
        with st.spinner(f"{label} 모델 불러오는 중..."):
            try:
                result = load_model_results(model_path, strategy_type)
                if strategy_type == "basic":
                    st.session_state.basic_results = result
                else:
                    st.session_state.tuned_results = result
                st.success(f"{label} 실행 완료!")
            except Exception as e:
                st.error(f"{label} 오류: {e}")

    # ── 시각화 ────────────────────────────────────────────────────
    b = st.session_state.basic_results
    t = st.session_state.tuned_results

    if b is None and t is None:
        st.info("모델을 실행하면 분석 결과가 여기에 표시됩니다.")
    else:
        st.markdown("---")

        # ── 1. 핵심 지표 카드 ──────────────────────────────────────
        st.markdown("## 핵심 성과 지표")

        metric_defs = [
            ("총 수익률 (%)", "total_return", "{:.2f}"),
            ("샤프 비율", "sharpe", "{:.4f}"),
            ("소르티노 비율", "sortino", "{:.4f}"),
            ("칼마 비율", "calmar", "{:.4f}"),
            ("최대 낙폭 (%)", "max_drawdown", lambda x: f"{x*100:.2f}"),
            ("승률 (%)", "win_rate", "{:.2f}"),
            ("최종 자산가치", "final_value", "{:.4f}"),
        ]

        cols = st.columns(len(metric_defs))
        for col, (label, key, fmt) in zip(cols, metric_defs):
            b_val = (fmt(b[key]) if callable(fmt) else fmt.format(b[key])) if b else "—"
            t_val = (fmt(t[key]) if callable(fmt) else fmt.format(t[key])) if t else "—"
            with col:
                st.metric(label=f"🔵 {label}", value=b_val)
                st.metric(label=f"🟢 {label}", value=t_val,
                          delta=None if (not b or not t) else None)

        # ── 2. 성과 지표 테이블 ────────────────────────────────────
        st.markdown("## 성과 지표 상세 비교")
        metrics_rows = []
        for res, name in [(b, "🔵 기본 전략"), (t, "🟢 최적화 전략")]:
            if res:
                metrics_rows.append({
                    "전략": name,
                    "총 수익률 (%)": f"{res['total_return']:.2f}",
                    "샤프 비율": f"{res['sharpe']:.4f}",
                    "소르티노 비율": f"{res['sortino']:.4f}",
                    "칼마 비율": f"{res['calmar']:.4f}",
                    "최대 낙폭 (%)": f"{res['max_drawdown']*100:.2f}",
                    "승률 (%)": f"{res['win_rate']:.2f}",
                    "최종 자산가치": f"{res['final_value']:.4f}",
                })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

        # ── 3. 누적 수익률 & 낙폭 ─────────────────────────────────
        st.markdown("## 누적 수익률 & 낙폭 분석")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("포트폴리오 자산가치 변화", "낙폭 (%)"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
        )

        if b:
            fig.add_trace(go.Scatter(
                y=b['portfolio_values'], mode='lines', name='기본 전략',
                line=dict(color=COLORS["basic"], width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                y=[-d * 100 for d in b['drawdowns']], mode='lines', name='기본 낙폭',
                line=dict(color=COLORS["basic_dd"], width=1.5),
                fill='tozeroy', fillcolor='rgba(214,39,40,0.2)'
            ), row=2, col=1)

        if t:
            fig.add_trace(go.Scatter(
                y=t['portfolio_values'], mode='lines', name='최적화 전략',
                line=dict(color=COLORS["tuned"], width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                y=[-d * 100 for d in t['drawdowns']], mode='lines', name='최적화 낙폭',
                line=dict(color=COLORS["tuned_dd"], width=1.5),
                fill='tozeroy', fillcolor='rgba(255,127,14,0.2)'
            ), row=2, col=1)

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.update_xaxes(title_text="거래일", row=2, col=1)
        fig.update_yaxes(title_text="포트폴리오 가치", row=1, col=1)
        fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
        fig.update_layout(height=700, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # ── 4. 수익률 분포 ────────────────────────────────────────
        st.markdown("## 일별 수익률 분포")
        col1, col2 = st.columns(2)

        with col1:
            fig_h = go.Figure()
            if b:
                fig_h.add_trace(go.Histogram(
                    x=b['returns'] * 100, name='기본 전략',
                    opacity=0.7, marker_color=COLORS["basic"], nbinsx=50
                ))
            if t:
                fig_h.add_trace(go.Histogram(
                    x=t['returns'] * 100, name='최적화 전략',
                    opacity=0.7, marker_color=COLORS["tuned"], nbinsx=50
                ))
            fig_h.update_layout(
                title="일별 수익률 히스토그램",
                xaxis_title="일별 수익률 (%)", yaxis_title="빈도",
                barmode='overlay', height=400
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col2:
            fig_b = go.Figure()
            if b:
                fig_b.add_trace(go.Box(
                    y=b['returns'] * 100, name='기본 전략',
                    marker_color=COLORS["basic"]
                ))
            if t:
                fig_b.add_trace(go.Box(
                    y=t['returns'] * 100, name='최적화 전략',
                    marker_color=COLORS["tuned"]
                ))
            fig_b.update_layout(
                title="수익률 박스플롯",
                yaxis_title="일별 수익률 (%)", height=400
            )
            st.plotly_chart(fig_b, use_container_width=True)

        # ── 5. 자산 배분 변화 + 평균 파이 (한 행) ─────────────────
        st.markdown("## 자산 배분 분석")

        # 두 전략 있으면 2열, 하나만 있으면 1열
        results_available = [(res, lbl) for res, lbl in [(b, "기본 전략"), (t, "최적화 전략")] if res]
        n_cols = len(results_available)
        weight_cols = st.columns(n_cols)

        for col, (res, lbl) in zip(weight_cols, results_available):
            with col:
                st.markdown(f"### {lbl}")
                weights_df = pd.DataFrame(res['weights'], columns=TICKERS)

                fig_w = go.Figure()
                for i, ticker in enumerate(TICKERS):
                    fig_w.add_trace(go.Scatter(
                        y=weights_df[ticker], mode='lines', name=ticker,
                        line=dict(width=2, color=ASSET_COLORS[i]),
                        stackgroup='one', groupnorm='percent'
                    ))
                fig_w.update_layout(
                    title="자산 비중 변화 (누적 %)",
                    xaxis_title="리밸런싱 시점", yaxis_title="비중 (%)",
                    yaxis=dict(ticksuffix="%"), height=400, hovermode='x unified'
                )
                st.plotly_chart(fig_w, use_container_width=True)

                # 파이 차트 (같은 열 바로 아래)
                avg_w = np.mean(res['weights'], axis=0)
                fig_p = go.Figure(data=[go.Pie(
                    labels=TICKERS, values=avg_w,
                    marker_colors=ASSET_COLORS, hole=0.3
                )])
                fig_p.update_layout(title=f"{lbl} 평균 배분", height=380)
                st.plotly_chart(fig_p, use_container_width=True)

        # ── 6. 롤링 샤프 비율 ────────────────────────────────────
        st.markdown("## 롤링 샤프 비율 (60일 이동평균)")

        def rolling_sharpe(returns, window=60):
            s = pd.Series(returns)
            return (s.rolling(window).mean() / s.rolling(window).std()) * np.sqrt(252)

        fig_rs = go.Figure()
        if b:
            fig_rs.add_trace(go.Scatter(
                y=rolling_sharpe(b['returns']), mode='lines',
                name='기본 전략', line=dict(color=COLORS["basic"], width=2)
            ))
        if t:
            fig_rs.add_trace(go.Scatter(
                y=rolling_sharpe(t['returns']), mode='lines',
                name='최적화 전략', line=dict(color=COLORS["tuned"], width=2)
            ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rs.update_layout(
            title="롤링 샤프 비율 (60일 윈도우)",
            xaxis_title="거래일", yaxis_title="샤프 비율",
            height=400, hovermode='x unified'
        )
        st.plotly_chart(fig_rs, use_container_width=True)

        # ── 7. 월별 수익률 히트맵 ────────────────────────────────
        if b or t:
            st.markdown("## 월별 수익률 히트맵")
            heat_cols = st.columns(n_cols)

            for col, (res, lbl) in zip(heat_cols, results_available):
                with col:
                    n_days = len(res['returns'])
                    start_date = pd.Timestamp("2024-01-01")
                    dates = pd.bdate_range(start=start_date, periods=n_days)
                    ret_series = pd.Series(res['returns'], index=dates)

                    monthly = ret_series.resample('ME').apply(
                        lambda x: (1 + x).prod() - 1
                    ) * 100
                    monthly_df = monthly.reset_index()
                    monthly_df.columns = ['date', 'return']
                    monthly_df['year'] = monthly_df['date'].dt.year
                    monthly_df['month'] = monthly_df['date'].dt.month

                    pivot = monthly_df.pivot(index='year', columns='month', values='return')
                    month_labels = ['1월','2월','3월','4월','5월','6월',
                                    '7월','8월','9월','10월','11월','12월']

                    fig_hm = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=[month_labels[m-1] for m in pivot.columns],
                        y=[str(y) for y in pivot.index],
                        colorscale='RdYlGn',
                        zmid=0,
                        text=np.round(pivot.values, 2),
                        texttemplate="%{text}%",
                        hovertemplate="%{y}년 %{x}: %{z:.2f}%<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        title=f"{lbl} 월별 수익률 (%)",
                        height=300
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — 직접 학습
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.header("처음부터 직접 학습하기")
    st.warning("각 전략당 약 3–5분 소요됩니다. 학습 완료 후 '빠른 실행' 탭에서 재사용 가능합니다.")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        train_all = st.button("두 전략 동시 학습", use_container_width=True,
                              type="primary", help="기본 + 최적화 전략을 순차 학습합니다 (총 6–10분)")
    with col2:
        train_basic_only = st.button("기본 전략만 학습", use_container_width=True)
        train_tuned_only = st.button("최적화 전략만 학습", use_container_width=True)

    def do_train(strategy_fn, key, label):
        with st.spinner(f"{label} 학습 중... (3–5분 소요)"):
            try:
                results = strategy_fn()
                pr = results['returns']
                extra = compute_extra_metrics(pr, results['total_return'], results['max_drawdown'])
                results.update(extra)
                st.session_state[key] = results
                st.success(f"{label} 학습 완료! '빠른 실행' 탭에서 바로 확인하세요.")
            except Exception as e:
                st.error(f"{label} 오류: {e}")

    if train_all or train_basic_only:
        do_train(run_strategy, "basic_results", "기본 전략")
    if train_all or train_tuned_only:
        do_train(run_tuned_strategy, "tuned_results", "최적화 전략")


# ────────────────────────────────────────────────────────────────────
# 푸터
# ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("강화학습 기반 퀀트 전략 분석 대시보드 | 교육 및 연구 목적")
