"""
utils/styles.py
Nassau Candy — CSS theme injection
"""
import streamlit as st

def inject_css():
    st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }

  .kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
    border: 1px solid #2e3650; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 12px; text-align: center;
  }
  .kpi-value  { font-size: 2.2rem; font-weight: 700; color: #7dd3fc; line-height:1.1; }
  .kpi-label  { font-size: 0.82rem; color: #94a3b8; margin-top: 4px;
                text-transform: uppercase; letter-spacing:.05em; }
  .kpi-delta  { font-size: 0.78rem; margin-top: 6px; }
  .delta-pos  { color: #4ade80; }
  .delta-neg  { color: #f87171; }

  .section-title {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #7dd3fc; padding-left: 12px;
    margin: 28px 0 16px 0;
  }

  .insight-box {
    background: #1e293b; border-left: 4px solid #7dd3fc;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 12px 0;
    font-size: .88rem; color: #cbd5e1; line-height: 1.6;
  }

  .dataframe thead tr th { background: #1e2533 !important; color: #7dd3fc !important; }
  .dataframe tbody tr:nth-child(even) { background: #1a1f2c !important; }
  .dataframe tbody tr:hover           { background: #252b3e !important; }
</style>
""", unsafe_allow_html=True)
