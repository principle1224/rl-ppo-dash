import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from strategy import run_strategy
from strategy_tuned import run_tuned_strategy
import os

st.set_page_config(page_title="PPO 강화학습 포트폴리오 최적화", page_icon="", layout="wide")

# 타이틀
st.title(" PPO 강화학습 포트폴리오 최적화 대시보드")
st.markdown("**기본 전략 vs Optuna 최적화 전략 성과 비교**")
st.markdown("**버튼 클릭 시 약 10초간 대기 후 남은 버튼 모두 클릭해야 비교분석 가능**")
st.markdown("---")

# 사이드바 설명
with st.sidebar:
    st.header(" 전략 개요")
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
    - 기간: 2005-2023 (19년)
    - 타임스텝: 300,000
    
    **테스트 기간**
    - 기간: 2024-2025 (2년)
    - 리밸런싱: 월 1회 (20일)
    """)
    
    st.markdown("---")
    st.caption(" 교육 및 연구 목적 | 투자 조언 아님")

# 탭 구성
tab1, tab2 = st.tabs([" 빠른 실행 (사전학습 모델)", " 직접 학습하기"])

# ==================== 탭1: 빠른 비교 ====================
with tab1:
    st.header(" 사전학습된 모델로 빠르게 결과 확인하기")
    st.info(" 이미 학습된 모델을 불러와서 2024-2025년 성과를 즉시 확인합니다 (약 5초 소요)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        run_basic = st.button("🔵 기본 전략 실행", use_container_width=True, type="primary", 
                             help="PPO 알고리즘 기본 설정으로 학습된 모델")
    
    with col2:
        run_tuned = st.button("🟢 최적화 전략 실행", use_container_width=True, type="primary",
                             help="Optuna로 하이퍼파라미터 최적화된 모델")
    
    # 세션 상태 초기화
    if 'basic_results' not in st.session_state:
        st.session_state.basic_results = None
    if 'tuned_results' not in st.session_state:
        st.session_state.tuned_results = None
    
    # 기본 전략 실행
    if run_basic:
        if os.path.exists("models/ray_dalio_portfolio_model.zip"):
            with st.spinner("기본 전략 모델 불러오는 중..."):
                try:
                    from stable_baselines3 import PPO
                    from strategy import fetch_data, PortfolioEnvMonthly
                    
                    test_df = fetch_data("2024-01-01", "2025-12-31")
                    test_env = PortfolioEnvMonthly(test_df)
                    model = PPO.load("models/ray_dalio_portfolio_model", env=test_env)
                    
                    obs, _ = test_env.reset()
                    portfolio_values = [test_env.portfolio_value]
                    weights_history = []
                    done = False
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        weights_history.append(action.copy())
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        done = terminated or truncated
                        portfolio_values.append(info["portfolio_value"])
                    
                    final_value = portfolio_values[-1]
                    total_return = (final_value - 1.0) * 100
                    portfolio_returns = np.array(test_env.portfolio_returns)
                    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)
                    max_drawdown = max(test_env.drawdowns) if test_env.drawdowns else 0
                    
                    calmar = (total_return / 100) / (max_drawdown + 1e-8)
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
                    sortino = (np.mean(portfolio_returns) / downside_std) * np.sqrt(252)
                    win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100
                    
                    st.session_state.basic_results = {
                        "portfolio_values": portfolio_values,
                        "total_return": total_return,
                        "sharpe": sharpe,
                        "max_drawdown": max_drawdown,
                        "final_value": final_value,
                        "returns": portfolio_returns,
                        "drawdowns": test_env.drawdowns,
                        "weights": np.array(weights_history),
                        "calmar": calmar,
                        "sortino": sortino,
                        "win_rate": win_rate
                    }
                    st.success(" 기본 전략 실행 완료!")
                except Exception as e:
                    st.error(f" 오류 발생: {e}")
        else:
            st.warning(" 사전학습된 모델이 없습니다. '직접 학습하기' 탭에서 먼저 학습을 진행해주세요.")
    
    # 튜닝 전략 실행
    if run_tuned:
        if os.path.exists("models/ray_dalio_tuned_model.zip"):
            with st.spinner("최적화 전략 모델 불러오는 중..."):
                try:
                    from stable_baselines3 import PPO
                    from strategy_tuned import fetch_data, PortfolioEnvMonthly
                    
                    test_df = fetch_data("2024-01-01", "2025-12-31")
                    test_env = PortfolioEnvMonthly(test_df)
                    model = PPO.load("models/ray_dalio_tuned_model", env=test_env)
                    
                    obs, _ = test_env.reset()
                    portfolio_values = [test_env.portfolio_value]
                    weights_history = []
                    done = False
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        weights_history.append(action.copy())
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        done = terminated or truncated
                        portfolio_values.append(info["portfolio_value"])
                    
                    final_value = portfolio_values[-1]
                    total_return = (final_value - 1.0) * 100
                    portfolio_returns = np.array(test_env.portfolio_returns)
                    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)
                    max_drawdown = max(test_env.drawdowns) if test_env.drawdowns else 0
                    
                    calmar = (total_return / 100) / (max_drawdown + 1e-8)
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
                    sortino = (np.mean(portfolio_returns) / downside_std) * np.sqrt(252)
                    win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100
                    
                    st.session_state.tuned_results = {
                        "portfolio_values": portfolio_values,
                        "total_return": total_return,
                        "sharpe": sharpe,
                        "max_drawdown": max_drawdown,
                        "final_value": final_value,
                        "returns": portfolio_returns,
                        "drawdowns": test_env.drawdowns,
                        "weights": np.array(weights_history),
                        "calmar": calmar,
                        "sortino": sortino,
                        "win_rate": win_rate
                    }
                    st.success(" 최적화 전략 실행 완료!")
                except Exception as e:
                    st.error(f" 오류 발생: {e}")
        else:
            st.warning(" 사전학습된 모델이 없습니다. '직접 학습하기' 탭에서 먼저 학습을 진행해주세요.")
    
    # ==================== 시각화 섹션 ====================
    if st.session_state.basic_results or st.session_state.tuned_results:
        st.markdown("---")
        
        # Section 1: 성과 지표 테이블
        st.markdown("##  성과 지표 비교")
        
        metrics_data = []
        if st.session_state.basic_results:
            b = st.session_state.basic_results
            metrics_data.append({
                "전략": "기본 전략",
                "총 수익률 (%)": f"{b['total_return']:.2f}",
                "샤프 비율": f"{b['sharpe']:.4f}",
                "소르티노 비율": f"{b['sortino']:.4f}",
                "칼마 비율": f"{b['calmar']:.4f}",
                "최대 낙폭 (%)": f"{b['max_drawdown']*100:.2f}",
                "승률 (%)": f"{b['win_rate']:.2f}",
                "최종 자산가치": f"{b['final_value']:.4f}"
            })
        
        if st.session_state.tuned_results:
            t = st.session_state.tuned_results
            metrics_data.append({
                "전략": "최적화 전략",
                "총 수익률 (%)": f"{t['total_return']:.2f}",
                "샤프 비율": f"{t['sharpe']:.4f}",
                "소르티노 비율": f"{t['sortino']:.4f}",
                "칼마 비율": f"{t['calmar']:.4f}",
                "최대 낙폭 (%)": f"{t['max_drawdown']*100:.2f}",
                "승률 (%)": f"{t['win_rate']:.2f}",
                "최종 자산가치": f"{t['final_value']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Section 2: 누적 수익률 & 낙폭
        if st.session_state.basic_results and st.session_state.tuned_results:
            st.markdown("##  누적 수익률 & 낙폭 분석")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("포트폴리오 자산가치 변화", "낙폭 (%)"),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )
            
            # 누적 수익률
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.basic_results['portfolio_values'],
                    mode='lines',
                    name='기본 전략',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.tuned_results['portfolio_values'],
                    mode='lines',
                    name='최적화 전략',
                    line=dict(color='#2ca02c', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
            
            # 낙폭
            fig.add_trace(
                go.Scatter(
                    y=[-d*100 for d in st.session_state.basic_results['drawdowns']],
                    mode='lines',
                    name='기본 전략 낙폭',
                    line=dict(color='#d62728', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(214, 39, 40, 0.2)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    y=[-d*100 for d in st.session_state.tuned_results['drawdowns']],
                    mode='lines',
                    name='최적화 전략 낙폭',
                    line=dict(color='#ff7f0e', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(255, 127, 14, 0.2)'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="거래일", row=2, col=1)
            fig.update_yaxes(title_text="포트폴리오 가치", row=1, col=1)
            fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
            
            fig.update_layout(height=700, hovermode='x unified', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Section 3: 수익률 분포
            st.markdown("##  일별 수익률 분포 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=st.session_state.basic_results['returns']*100,
                    name='기본 전략',
                    opacity=0.7,
                    marker_color='#1f77b4',
                    nbinsx=50
                ))
                fig.add_trace(go.Histogram(
                    x=st.session_state.tuned_results['returns']*100,
                    name='최적화 전략',
                    opacity=0.7,
                    marker_color='#2ca02c',
                    nbinsx=50
                ))
                fig.update_layout(
                    title="일별 수익률 히스토그램",
                    xaxis_title="일별 수익률 (%)",
                    yaxis_title="빈도",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=st.session_state.basic_results['returns']*100,
                    name='기본 전략',
                    marker_color='#1f77b4'
                ))
                fig.add_trace(go.Box(
                    y=st.session_state.tuned_results['returns']*100,
                    name='최적화 전략',
                    marker_color='#2ca02c'
                ))
                fig.update_layout(
                    title="수익률 박스플롯",
                    yaxis_title="일별 수익률 (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Section 4: 자산 배분 변화
            st.markdown("##  자산 배분 변화 추이")
            
            tickers = ["SPY", "TLT", "GLD", "DBC", "SHY"]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 기본 전략")
                weights_df = pd.DataFrame(
                    st.session_state.basic_results['weights'],
                    columns=tickers
                )
                
                fig = go.Figure()
                for i, ticker in enumerate(tickers):
                    fig.add_trace(go.Scatter(
                        y=weights_df[ticker],
                        mode='lines',
                        name=ticker,
                        line=dict(width=2, color=colors[i]),
                        stackgroup='one',
                        groupnorm='percent'
                    ))
                
                fig.update_layout(
                    title="자산 비중 변화 (누적 %)",
                    xaxis_title="리밸런싱 시점",
                    yaxis_title="비중 (%)",
                    yaxis=dict(ticksuffix="%"),
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 최적화 전략")
                weights_df = pd.DataFrame(
                    st.session_state.tuned_results['weights'],
                    columns=tickers
                )
                
                fig = go.Figure()
                for i, ticker in enumerate(tickers):
                    fig.add_trace(go.Scatter(
                        y=weights_df[ticker],
                        mode='lines',
                        name=ticker,
                        line=dict(width=2, color=colors[i]),
                        stackgroup='one',
                        groupnorm='percent'
                    ))
                
                fig.update_layout(
                    title="자산 비중 변화 (누적 %)",
                    xaxis_title="리밸런싱 시점",
                    yaxis_title="비중 (%)",
                    yaxis=dict(ticksuffix="%"),
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Section 5: 평균 자산 배분
            st.markdown("## 평균 자산 배분 비교")
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_weights_basic = np.mean(st.session_state.basic_results['weights'], axis=0)
                fig = go.Figure(data=[go.Pie(
                    labels=tickers,
                    values=avg_weights_basic,
                    marker_colors=colors,
                    hole=0.3
                )])
                fig.update_layout(title="기본 전략 평균 배분", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_weights_tuned = np.mean(st.session_state.tuned_results['weights'], axis=0)
                fig = go.Figure(data=[go.Pie(
                    labels=tickers,
                    values=avg_weights_tuned,
                    marker_colors=colors,
                    hole=0.3
                )])
                fig.update_layout(title="최적화 전략 평균 배분", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Section 6: 롤링 샤프 비율
            st.markdown("##  롤링 샤프 비율 (60일 이동평균)")
            
            window = 60
            
            def rolling_sharpe(returns, window):
                rolling_mean = pd.Series(returns).rolling(window).mean()
                rolling_std = pd.Series(returns).rolling(window).std()
                return (rolling_mean / rolling_std) * np.sqrt(252)
            
            basic_rolling_sharpe = rolling_sharpe(st.session_state.basic_results['returns'], window)
            tuned_rolling_sharpe = rolling_sharpe(st.session_state.tuned_results['returns'], window)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=basic_rolling_sharpe,
                mode='lines',
                name='기본 전략',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=tuned_rolling_sharpe,
                mode='lines',
                name='최적화 전략',
                line=dict(color='#2ca02c', width=2)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                title=f"롤링 샤프 비율 ({window}일 윈도우)",
                xaxis_title="거래일",
                yaxis_title="샤프 비율",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== 탭2: 직접 학습 ====================
with tab2:
    st.header(" 처음부터 직접 학습하기")
    st.warning(" 각 전략당 약 3-5분 소요됩니다. 학습이 완료되면 '빠른 실행' 탭에서 재사용할 수 있습니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔵 기본 전략 학습 시작", use_container_width=True, help="2005-2023년 데이터로 PPO 기본 설정 학습"):
            with st.spinner("기본 전략 학습 중... (3-5분 소요)"):
                try:
                    results = run_strategy()
                    
                    portfolio_returns = results['returns']
                    
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
                    sortino = (np.mean(portfolio_returns) / downside_std) * np.sqrt(252)
                    calmar = (results['total_return'] / 100) / (results['max_drawdown'] + 1e-8)
                    win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100
                    
                    results['sortino'] = sortino
                    results['calmar'] = calmar
                    results['win_rate'] = win_rate
                    
                    st.session_state.basic_results = results
                    st.success(" 기본 전략 학습 완료!")
                    st.info(" 모델이 저장되었습니다. 이제 '빠른 실행' 탭에서 사용할 수 있습니다.")
                except Exception as e:
                    st.error(f" 오류 발생: {e}")
    
    with col2:
        if st.button("🟢 최적화 전략 학습 시작", use_container_width=True, help="Optuna로 최적화된 하이퍼파라미터로 학습"):
            with st.spinner("최적화 전략 학습 중... (3-5분 소요)"):
                try:
                    results = run_tuned_strategy()
                    
                    portfolio_returns = results['returns']
                    
                    downside_returns = portfolio_returns[portfolio_returns < 0]
                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
                    sortino = (np.mean(portfolio_returns) / downside_std) * np.sqrt(252)
                    calmar = (results['total_return'] / 100) / (results['max_drawdown'] + 1e-8)
                    win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) * 100
                    
                    results['sortino'] = sortino
                    results['calmar'] = calmar
                    results['win_rate'] = win_rate
                    
                    st.session_state.tuned_results = results
                    st.success(" 최적화 전략 학습 완료!")
                    st.info(" 모델이 저장되었습니다. 이제 '빠른 실행' 탭에서 사용할 수 있습니다.")
                except Exception as e:
                    st.error(f" 오류 발생: {e}")

# 푸터
st.markdown("---")
st.caption(" 강화학습 기반 퀀트 전략 분석 대시보드")
st.caption(" 교육 및 연구 목적")