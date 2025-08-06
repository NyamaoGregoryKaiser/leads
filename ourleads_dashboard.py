import streamlit as st
import pandas as pd
import gspread
from datetime import datetime
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import json
import os
from google.oauth2.service_account import Credentials

# Force light mode for the app
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #fff !important;
        color: #222 !important;
    }
    [data-testid="stHeader"] {
        background: #fff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import matplotlib as mpl
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '#222',
    'axes.labelcolor': '#222',
    'xtick.color': '#222',
    'ytick.color': '#222',
    'text.color': '#222',
    'axes.titlecolor': '#222',
    'grid.color': '#ccc',
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.prop_cycle': mpl.cycler(color=['#0074D9', '#FF4136', '#2ECC40', '#FF851B', '#B10DC9'])
})

st.set_page_config(page_title="Our Leads Dashboard", layout="wide")

# --- Google Sheets Setup ---
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1CNR-5kLURpyEjbKVqA_ejtzgSdDsXMD9EmWtO7t2Ml0/edit?usp=sharing'
SHEET_ID = SHEET_URL.split('/')[5]

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]

@st.cache_data(ttl=20)
def get_leads_dataframe():
    try:
        if 'GOOGLE_CREDENTIALS' in os.environ:
            creds_json = os.environ['GOOGLE_CREDENTIALS']
            creds_json = creds_json.strip().strip("'").strip('"')
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
            # st.success("✅ Using credentials from environment variables")
        else:
            if os.path.exists('credentials.json'):
                creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
                # st.success("✅ Using local credentials.json file")
            else:
                st.error("❌ No credentials found!")
                st.info("For local development: Add credentials.json file")
                st.info("For Streamlit Cloud: Add GOOGLE_CREDENTIALS to secrets")
                return pd.DataFrame()
        
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        # st.success(f"✅ Successfully loaded {len(df)} records from Google Sheets")
        return df
    except Exception as e:
        st.error(f"❌ Error connecting to Google Sheets: {str(e)}")
        st.info("Please check your credentials and internet connection.")
        return pd.DataFrame()

# --- Auto-refresh logic ---
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()

if time.time() - st.session_state['last_refresh'] > 20:
    st.cache_data.clear()  # Clear the cache to force a refresh
    st.session_state['last_refresh'] = time.time()

# Add this for auto-rerun every 20 seconds
if 'last_autorefresh' not in st.session_state:
    st.session_state['last_autorefresh'] = time.time()
if time.time() - st.session_state['last_autorefresh'] > 20:
    st.session_state['last_autorefresh'] = time.time()
    st.rerun()

st.markdown("""
    <style>
    html, body, .main, .block-container, .appview-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    header, [data-testid="stToolbar"] {
        display: none !important;
    }
    .main .block-container > :first-child {
        margin-top: -4rem !important;
    }
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .big-font {font-size:40px !important; font-weight:bold; color:#00FFD0;}
    .card-label {font-size:20px !important; color:white; font-weight:bold;}
    .stMetric {background-color: #181818; border-radius: 10px; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

# Add logo and title in a horizontal row using Streamlit columns
# Three-column layout: logo left, title centered, empty right
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0; margin-top:0; font-size:2.5rem; font-weight:bold; color:#222;'>Phoenix Leads Dashboard</h1>",
        unsafe_allow_html=True
    )

# Load data
df = get_leads_dataframe()

# Check if data was loaded successfully
if df.empty:
    st.error("Unable to load data from Google Sheets. Please check your credentials and try again.")
    st.stop()

# Parse dates
if 'DATE' in df.columns:
    df['DATE_parsed'] = pd.to_datetime(df['DATE'], errors='coerce')
else:
    st.error("No DATE column found in the data. Please check your Google Sheet structure.")
    st.stop()
    df['DATE_parsed'] = pd.NaT

# Update custom CSS for wider selectboxes
st.markdown('''
    <style>
    .small-filter .stSelectbox > div[data-baseweb="select"] {
        font-size: 12px !important;
        min-height: 28px !important;
        height: 28px !important;
        max-width: 360px;
    }
    .small-filter label { font-size: 12px !important; }
    </style>
''', unsafe_allow_html=True)

# Branch filter for summary cards (top-right above cards)
_, filter_col_cards = st.columns([10, 1])
with filter_col_cards:
    with st.container():
        st.markdown('<div class="small-filter">', unsafe_allow_html=True)
        branches_cards = ['All'] + sorted(df['BRANCH'].unique()) if 'BRANCH' in df.columns else ['All']
        selected_branch_cards = st.selectbox('Filter by Branch (Cards)', branches_cards, key='branch_filter_cards')
        st.markdown('</div>', unsafe_allow_html=True)
if selected_branch_cards != 'All' and 'BRANCH' in df.columns:
    df_cards = df[df['BRANCH'] == selected_branch_cards].copy()
else:
    df_cards = df.copy()

# Recalculate card metrics based on filtered df_cards
today = pd.Timestamp(datetime.now().date())
leads_today = df_cards[df_cards['DATE_parsed'].dt.date == today.date()].shape[0]
yesterday = today - pd.Timedelta(days=1)
leads_yesterday = df_cards[df_cards['DATE_parsed'].dt.date == yesterday.date()].shape[0]
leads_today_delta = leads_today - leads_yesterday
today_arrow = f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if leads_today_delta > 0 else (f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if leads_today_delta < 0 else '')
if leads_yesterday > 0:
    today_pct = (leads_today_delta / leads_yesterday) * 100
    today_pct_str = f'<span style="color:{"#2ECC40" if leads_today_delta > 0 else "#FF4136"}; font-size:14px; font-weight:bold;">{abs(today_pct):.1f}%</span>'
else:
    today_pct_str = ''
week_start = today - pd.Timedelta(days=today.weekday())
days_so_far = (today - week_start).days + 1
leads_week = df_cards[(df_cards['DATE_parsed'] >= week_start) & (df_cards['DATE_parsed'] <= today)].shape[0]
prev_week_start = week_start - pd.Timedelta(days=7)
prev_week_end = prev_week_start + pd.Timedelta(days=days_so_far - 1)
leads_prev_week = df_cards[(df_cards['DATE_parsed'] >= prev_week_start) & (df_cards['DATE_parsed'] <= prev_week_end)].shape[0]
week_delta = leads_week - leads_prev_week
week_arrow = (
    f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if week_delta > 0 else
    f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if week_delta < 0 else ''
)
if leads_prev_week > 0:
    week_pct = (week_delta / leads_prev_week) * 100
    week_pct_str = f'<span style="color:{"#2ECC40" if week_delta > 0 else "#FF4136"}; font-size:14px; font-weight:bold;">{abs(week_pct):.1f}%</span>'
else:
    week_pct_str = ''
month_start = today.replace(day=1)
days_so_far_month = today.day
leads_month = df_cards[(df_cards['DATE_parsed'] >= month_start) & (df_cards['DATE_parsed'] <= today)].shape[0]
prev_month_end = month_start - pd.Timedelta(days=1)
prev_month_start = prev_month_end.replace(day=1)
prev_month_same_day = prev_month_start + pd.Timedelta(days=days_so_far_month - 1)
leads_prev_month = df_cards[(df_cards['DATE_parsed'] >= prev_month_start) & (df_cards['DATE_parsed'] <= prev_month_same_day)].shape[0]
month_delta = leads_month - leads_prev_month
month_arrow = (
    f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if month_delta > 0 else
    f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if month_delta < 0 else ''
)
if leads_prev_month > 0:
    month_pct = (month_delta / leads_prev_month) * 100
    month_pct_str = f'<span style="color:{"#2ECC40" if month_delta > 0 else "#FF4136"}; font-size:14px; font-weight:bold;">{abs(month_pct):.1f}%</span>'
else:
    month_pct_str = ''
total_leads = len(df_cards)

# Responsive flexbox for cards
st.markdown('''
    <style>
    .card-row-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.2em;
        overflow-x: auto;
        margin-bottom: 1em;
        max-width: 100%;
        width: 100%;
        justify-content: center;
    }
    .card-row-metrics .dashboard-card {
        flex: 1 0 120px;
        min-width: 100px;
        max-width: 220px;
        margin-bottom: 0.3em;
        box-sizing: border-box;
    }
    @media (max-width: 600px) {
        .card-row-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.2em;
            width: 100%;
        }
        .card-row-metrics .dashboard-card {
            width: 100%;
            min-width: 0;
            max-width: 100%;
            margin-bottom: 0.3em;
            box-sizing: border-box;
        }
    }
    </style>
''', unsafe_allow_html=True)

# Build the card HTML
card_style = """
    background: #F7F7F9;
    border-radius: 10px;
    box-shadow: 0 2px 12px 0 rgba(0,0,0,0.08);
    border: 2px solid #0074D9;
    padding: 0.18em 0.15em 0.12em 0.15em;
    margin: 0.18em 0.1em 0.18em 0.1em;
    text-align: center;
    min-width: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1;
    box-sizing: border-box;
"""

st.markdown('''
    <style>
    .dashboard-card {
        min-height: 60px !important;
    }
    @media (max-width: 600px) {
        .dashboard-card {
            min-height: 32px !important;
        }
    }
    </style>
''', unsafe_allow_html=True)
number_style = "font-size: 16px; font-weight: bold; color: #001F3F; text-shadow: none; margin-bottom: 0.02em;"
label_style = "font-size: 9px; color: #555; font-weight: bold; letter-spacing: 0.2px; text-shadow: none;"

card_html = f'''
<div class="card-row-metrics">
  <div class="dashboard-card" style="{card_style}">
    <div style="{number_style}">{total_leads}</div>
    <div style="{label_style}">Total Leads</div>
  </div>
  <div class="dashboard-card" style="{card_style}">
    <div style="{number_style}">{leads_today} {today_arrow} {today_pct_str}</div>
    <div style="{label_style}">Leads Today</div>
  </div>
  <div class="dashboard-card" style="{card_style}">
    <div style="{number_style}">{leads_week} {week_arrow} {week_pct_str}</div>
    <div style="{label_style}">Leads This Week</div>
  </div>
  <div class="dashboard-card" style="{card_style}">
    <div style="{number_style}">{leads_month} {month_arrow} {month_pct_str}</div>
    <div style="{label_style}">Leads This Month</div>
  </div>
</div>
'''
st.markdown(card_html, unsafe_allow_html=True)

st.markdown("---")

# Remove or comment out the global conversion to string
# if not df.empty:
#     df = df.astype(str)
# If you need to display a table as strings, use:
# display_df = df.astype(str)
# st.dataframe(display_df)
# --- Table ---

# --- Counts ---
col_branch, col_source = st.columns([1.5, 1.2])
with col_branch:
    # Inline header and filter for Leads per Branch and Source
    col_header_branch, col_filter_branch = st.columns([2, 1])
    with col_header_branch:
        st.subheader("Leads per Branch and Source", anchor=False)
    with col_filter_branch:
        st.markdown('<div class="small-filter">', unsafe_allow_html=True)
        months_branch = ['All'] + sorted(df['DATE_parsed'].dropna().dt.strftime('%Y-%m').unique()) if 'DATE_parsed' in df.columns else ['All']
        selected_month_branch = st.selectbox(' ', months_branch, key='month_filter_branch_source')
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply month filter
    if selected_month_branch != 'All' and 'DATE_parsed' in df.columns:
        df_branch_source = df[df['DATE_parsed'].dt.strftime('%Y-%m') == selected_month_branch]
    else:
        df_branch_source = df
    
    if 'BRANCH' in df_branch_source.columns and 'SOURCE OF LEAD GENERATION' in df_branch_source.columns:
        # Create a pivot table: rows=BRANCH, columns=SOURCE OF LEAD GENERATION, values=count
        branch_source = pd.pivot_table(df_branch_source, index='BRANCH', columns='SOURCE OF LEAD GENERATION', aggfunc='size', fill_value=0)
        import matplotlib.pyplot as plt
        branch_source = branch_source.loc[branch_source.sum(axis=1).sort_values(ascending=False).index]  # Sort branches by total leads
        fig, ax = plt.subplots(figsize=(10, max(6, 0.5*len(branch_source))))
        branch_source.plot(kind='barh', stacked=True, ax=ax, colormap='tab20')
        ax.set_xlabel('Leads', fontsize=12)
        ax.set_ylabel('Branch', fontsize=12)
        ax.set_title('')
        ax.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info('BRANCH or SOURCE OF LEAD GENERATION column not found in the data.')
with col_source:
    # Inline header and filter for Leads per Source
    col_header, col_filter = st.columns([2, 1])
    with col_header:
        st.subheader("Leads per Source", anchor=False)
    with col_filter:
        st.markdown('<div class="small-filter">', unsafe_allow_html=True)
        branches = ['All'] + sorted(df['BRANCH'].unique()) if 'BRANCH' in df.columns else ['All']
        selected_branch = st.selectbox(' ', branches, key='branch_filter_source')
        st.markdown('</div>', unsafe_allow_html=True)
    if selected_branch != 'All' and 'BRANCH' in df.columns:
        df_source = df[df['BRANCH'] == selected_branch]
    else:
        df_source = df
    source_counts = df_source['SOURCE OF LEAD GENERATION'].value_counts()
    import plotly.graph_objects as go
    bar_fig = go.Figure()
    bar_fig.add_bar(x=source_counts.index, y=source_counts.values, marker_color="#0074D9")
    bar_fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color='#222', size=14),
        xaxis=dict(
            title=dict(text='Source', font=dict(color='#222', size=16)),
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc'
        ),
        yaxis=dict(
            title=dict(text='Leads', font=dict(color='#222', size=16)),
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc'
        ),
        margin=dict(l=20, r=20, t=30, b=40),
        showlegend=False
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})

# Place Leads per Team Lead and Team Leader Conversion Ratios side by side
col_leads, col_conv = st.columns(2)
with col_leads:
    # Inline header and filter for Leads per Team Lead
    col_header_team, col_filter_team = st.columns([2, 1])
    with col_header_team:
        st.subheader("Leads per Team Lead", anchor=False)
    with col_filter_team:
        st.markdown('<div class="small-filter">', unsafe_allow_html=True)
        months_team = ['All'] + sorted(df['DATE_parsed'].dropna().dt.strftime('%Y-%m').unique()) if 'DATE_parsed' in df.columns else ['All']
        selected_month_team = st.selectbox(' ', months_team, key='month_filter_team_lead')
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply month filter
    if selected_month_team != 'All' and 'DATE_parsed' in df.columns:
        df_team = df[df['DATE_parsed'].dt.strftime('%Y-%m') == selected_month_team]
    else:
        df_team = df
    
    team_lead_counts = df_team['TEAM LEADER NAME'].value_counts()
    bar_fig2 = go.Figure()
    bar_fig2.add_bar(x=team_lead_counts.index, y=team_lead_counts.values, marker_color="#0074D9")
    bar_fig2.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color='#222', size=14),
        xaxis=dict(
            title=dict(text='Team Lead', font=dict(color='#222', size=16)),
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc'
        ),
        yaxis=dict(
            title=dict(text='Leads', font=dict(color='#222', size=16)),
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc'
        ),
        margin=dict(l=20, r=20, t=30, b=40),
        showlegend=False
    )
    st.plotly_chart(bar_fig2, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
with col_conv:
    # --- Conversion Ratio Analysis ---
    st.subheader("Team Leader Conversion Ratios (June & July Sales)")
    # (Insert the entire conversion ratio calculation and plotting code here, replacing the previous standalone block)
    # Load sales data
    df_sales = pd.read_csv('disbursement_report-june,july.csv', dtype=str)
    def normalize_contact(val):
        if pd.isnull(val):
            return []
        vals = str(val).replace('/', ',').replace(';', ',').split(',')
        normed = []
        for v in vals:
            v = v.strip().replace(' ', '').replace('+254', '0')
            if v.startswith('254'):
                v = '0' + v[3:]
            if v.startswith('0') and len(v) >= 10:
                normed.append(v[-9:])
            elif v.isdigit() and len(v) >= 9:
                normed.append(v[-9:])
        return normed
    sales_contacts = set()
    for col in ['Borrower Landline', 'Borrower Mobile']:
        if col in df_sales.columns:
            df_sales[col] = df_sales[col].astype(str)
            for contacts in df_sales[col].apply(normalize_contact):
                sales_contacts.update(contacts)
    if 'CLIENT CONTACT' in df.columns and 'TEAM LEADER NAME' in df.columns:
        bad_names = ["undefined", "none", "null", "nan"]
        df_valid = df[df['TEAM LEADER NAME'].notna()].copy()
        df_valid['TLN_CLEAN'] = df_valid['TEAM LEADER NAME'].str.strip().str.lower()
        for bad in bad_names:
            df_valid = df_valid[~df_valid['TLN_CLEAN'].str.contains(bad)]
        df_valid = df_valid[df_valid['TLN_CLEAN'] != ""]
        df_valid['CONTACT_NORM'] = df_valid['CLIENT CONTACT'].apply(lambda x: normalize_contact(x)[0] if normalize_contact(x) else np.nan)
        total_leads = df_valid.groupby('TEAM LEADER NAME').size()
        df_valid['converted'] = df_valid['CONTACT_NORM'].apply(lambda x: x in sales_contacts if pd.notnull(x) else False)
        converted_leads = df_valid[df_valid['converted']].groupby('TEAM LEADER NAME').size()
        ratio = (converted_leads / total_leads).fillna(0).sort_values(ascending=False)
        mask = ratio.index.str.strip().str.lower() != 'undefined'
        ratio = ratio[mask]
        
        # Ensure we have data to plot
        if len(ratio) > 0:
            import plotly.graph_objects as go
            team_leaders = ratio.index.tolist()
            conversion_perc = (ratio.values * 100).round(1)
            
            fig = go.Figure()
            fig.add_bar(
                x=conversion_perc,
                y=team_leaders,
                orientation='h',
                marker_color='#0074D9',
                hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
            )
            fig.update_layout(
                xaxis=dict(
                    title=dict(text='Conversion Ratio (%)', font=dict(color='#222', size=16)),
                    range=[0, max(15, conversion_perc.max() + 2)],
                    tickformat='.0f',
                    color='#222',
                    linecolor='#222',
                    tickfont=dict(color='#222'),
                    gridcolor='#ccc',
                    zerolinecolor='#ccc'
                ),
                yaxis=dict(
                    title=dict(text='Team Leader', font=dict(color='#222', size=16)),
                    color='#222',
                    linecolor='#222',
                    tickfont=dict(color='#222'),
                    gridcolor='#ccc',
                    zerolinecolor='#ccc',
                    # Explicitly set the category order to match the data
                    categoryorder='array',
                    categoryarray=team_leaders[::-1]  # Reverse to show highest at top
                ),
                margin=dict(l=20, r=20, t=40, b=40),
                width=650, height=350,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#222', size=14),
                title=None,
            )
            fig.update_xaxes(showgrid=True, gridcolor='#eee')
        else:
            st.info('No conversion data available for the selected filters.')
        st.markdown('<div style="overflow-x:auto;width=100%">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False, 'scrollZoom': True})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('CLIENT CONTACT or TEAM LEADER NAME column not found in the leads data.')

# After the cards and before the branch/source charts
st.markdown("---")

# --- Time Series Chart ---
header_col, filter_col_branch, filter_col_month = st.columns([3, 1, 1])
with header_col:
    st.subheader("Leads per Day - Time Series", anchor=False)
# Parse DATE_parsed ONCE at the start
if 'DATE' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['DATE_parsed']):
    df['DATE_parsed'] = pd.to_datetime(df['DATE'], errors='coerce')
# Branch filter
with filter_col_branch:
    branches_ts = ['All'] + sorted(df['BRANCH'].unique()) if 'BRANCH' in df.columns else ['All']
    selected_branch_ts = st.selectbox('Branch', branches_ts, key='branch_filter_timeseries')
# Filter for branch
if selected_branch_ts != 'All' and 'BRANCH' in df.columns:
    df_branch = df[df['BRANCH'] == selected_branch_ts]
else:
    df_branch = df
# Month filter options based on branch
if 'DATE_parsed' in df_branch.columns:
    months = ['All'] + sorted(df_branch['DATE_parsed'].dropna().dt.strftime('%Y-%m').unique())
else:
    months = ['All']
if 'month_filter_timeseries' in st.session_state:
    if st.session_state['month_filter_timeseries'] not in months:
        st.session_state['month_filter_timeseries'] = 'All'
with filter_col_month:
    selected_month_ts = st.selectbox('Month', months, key='month_filter_timeseries')
# Apply both filters
# Always apply month filter to branch-filtered DataFrame
df_ts = df_branch.copy()
if selected_month_ts != 'All' and 'DATE_parsed' in df_ts.columns:
    df_ts = df_ts[df_ts['DATE_parsed'].dt.strftime('%Y-%m') == selected_month_ts]
# Only plot if there is data
if not df_ts.empty and 'DATE_parsed' in df_ts.columns:
    daily_leads = (
        df_ts.dropna(subset=['DATE_parsed'])
        .groupby(df_ts['DATE_parsed'].dt.date)
        .size()
        .reset_index(name='leads_count')
    )
    daily_leads['date'] = pd.to_datetime(daily_leads['DATE_parsed']) if 'DATE_parsed' in daily_leads.columns else pd.to_datetime(daily_leads['date'])
    daily_leads = daily_leads.sort_values('date')
    # Create time series chart using plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = go.Figure()
    # Add line chart
    fig.add_trace(go.Scatter(
        x=daily_leads['date'],
        y=daily_leads['leads_count'],
        mode='lines+markers',
        name='Leads per Day',
        line=dict(color='#0074D9', width=3),
        marker=dict(size=6, color='#0074D9'),
        hovertemplate='Date: %{x}<br>Leads: %{y}<extra></extra>'
    ))
    # Add area fill
    fig.add_trace(go.Scatter(
        x=daily_leads['date'],
        y=daily_leads['leads_count'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 116, 217, 0.1)',
        line=dict(width=0),
        showlegend=False,
        hovertemplate='Date: %{x}<br>Leads: %{y}<extra></extra>'
    ))
    # Add trend line (linear regression)
    if len(daily_leads) > 1:
        import numpy as np
        x = np.arange(len(daily_leads))
        y = daily_leads['leads_count'].values
        # Fit linear regression
        coef = np.polyfit(x, y, 1)
        trend = np.poly1d(coef)(x)
        fig.add_trace(go.Scatter(
            x=daily_leads['date'],
            y=trend,
            mode='lines',
            name='Trend',
            line=dict(color='orange', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    # Update layout
    fig.update_layout(
        title=None,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#222', size=14),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=40),
        height=400,
        xaxis=dict(
            title=dict(text='Date', font=dict(color='#222', size=16)),
            showgrid=True,
            gridcolor='#eee',
            tickformat='%b %d, %Y',
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            zerolinecolor='#ccc'
        ),
        yaxis=dict(
            title=dict(text='Number of Leads', font=dict(color='#222', size=16)),
            showgrid=True,
            gridcolor='#eee',
            zeroline=True,
            zerolinecolor='#ccc',
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222')
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
    # Add summary statistics with responsive grid
    st.markdown('''
        <style>
        .summary-card-row-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.2em;
            overflow-x: auto;
            margin-bottom: 1em;
            max-width: 100%;
            width: 100%;
            justify-content: center;
        }
        .summary-card-row-metrics .dashboard-card {
            flex: 1 0 120px;
            min-width: 100px;
            max-width: 220px;
            margin-bottom: 0.3em;
            box-sizing: border-box;
        }
        @media (max-width: 600px) {
            .summary-card-row-metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.2em;
                width: 100%;
            }
            .summary-card-row-metrics .dashboard-card {
                width: 100%;
                min-width: 0;
                max-width: 100%;
                margin-bottom: 0.3em;
                box-sizing: border-box;
            }
        }
        </style>
    ''', unsafe_allow_html=True)
    avg_daily = daily_leads['leads_count'].mean()
    max_daily = daily_leads['leads_count'].max()
    max_date = daily_leads.loc[daily_leads['leads_count'].idxmax(), 'date']
    min_daily = daily_leads['leads_count'].min()
    min_date = daily_leads.loc[daily_leads['leads_count'].idxmin(), 'date']
    total_days = len(daily_leads)
    summary_card_html = f'''
    <div class="summary-card-row-metrics">
      <div class="dashboard-card" style="{card_style}">
        <div style="{number_style}">{avg_daily:.1f}</div>
        <div style="{label_style}">Average Daily Leads</div>
      </div>
      <div class="dashboard-card" style="{card_style}">
        <div style="{number_style}">{max_daily}</div>
        <div style="{label_style}">Peak Day ({max_date.strftime('%b %d')})</div>
      </div>
      <div class="dashboard-card" style="{card_style}">
        <div style="{number_style}">{min_daily}</div>
        <div style="{label_style}">Lowest Day ({min_date.strftime('%b %d')})</div>
      </div>
      <div class="dashboard-card" style="{card_style}">
        <div style="{number_style}">{total_days}</div>
        <div style="{label_style}">Days with Data</div>
      </div>
    </div>
    '''
    st.markdown(summary_card_html, unsafe_allow_html=True)
else:
    st.info('No data for the selected filters.')

st.markdown("---")

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Agents per Team Lead")
    # Team lead agent data
    team_lead_agents_data = {
        'CECILIA NAISORA MACKSON': 3,
        'Chris Konzi Mbinda': 12,
        'Cosmas Gathogo Njogu': 7,  # Combined the two entries (6 + 1)
        'ERIC OWUOR': 3,
        'George Barasa Barasa': 5,
        'GRACE WANGARI KARIUKI': 6,
        'Irene Awuor': 5,
        'Josephine Ndewa Kitonga': 3,
        'Judy Gathoni Wangu': 3,
        'Kevin Ouma Odongo': 5,
        'Lydia Njora Njora': 1,
        'Morris Maina Gichuki': 5,
        'Peter Mosingo Matuyia': 5,
        'PETER OWUOR OUMA': 4,
        'Teresia Wambui Nyambura': 1
    }
    
    # Convert to pandas Series and sort by number of agents
    team_lead_agents = pd.Series(team_lead_agents_data).sort_values(ascending=False)
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(
        x=team_lead_agents.values,
        y=team_lead_agents.index,
        orientation='h',
        marker_color='#0074D9',
        hovertemplate='%{y}: %{x} agents<extra></extra>'
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Number of Agents', font=dict(color='#222', size=16)),
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc',
        ),
        yaxis=dict(
            title=dict(text='Team Lead', font=dict(color='#222', size=16)),
            autorange="reversed",
            color='#222',
            linecolor='#222',
            tickfont=dict(color='#222'),
            gridcolor='#ccc',
            zerolinecolor='#ccc',
        ),
        margin=dict(l=20, r=20, t=30, b=40),
        width=650, height=350,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#222', size=14),
        title=None,
    )
    st.markdown('<div style="overflow-x:auto;width=100%">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False, 'scrollZoom': True})
    st.markdown('</div>', unsafe_allow_html=True)
with col_right:
    st.subheader("Top Locations")
    if 'LOCATION' in df.columns:
        location_counts = df['LOCATION'].value_counts().head(10)
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.viridis(np.linspace(0, 1, len(location_counts)))
        ax.barh(location_counts.index[::-1], location_counts.values[::-1], color=colors)
        ax.set_xlabel('Leads')
        ax.set_ylabel('Location')
        ax.set_title('')
        # Remove the frame (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)
    else:
        st.info('No LOCATION column found in the data.') 