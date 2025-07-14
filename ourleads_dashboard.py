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

today = pd.Timestamp(datetime.now().date())
leads_today = df[df['DATE_parsed'].dt.date == today.date()].shape[0]
yesterday = today - pd.Timedelta(days=1)
leads_yesterday = df[df['DATE_parsed'].dt.date == yesterday.date()].shape[0]
# Calculate delta and arrow for today vs yesterday
leads_today_delta = leads_today - leads_yesterday
today_arrow = f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if leads_today_delta > 0 else (f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if leads_today_delta < 0 else '')
if leads_yesterday > 0:
    today_pct = (leads_today_delta / leads_yesterday) * 100
    today_pct_str = f'<span style="color:{"#2ECC40" if leads_today_delta > 0 else "#FF4136"}; font-size:20px; font-weight:bold;">{abs(today_pct):.1f}%</span>'
else:
    today_pct_str = ''
week_start = today - pd.Timedelta(days=today.weekday())
leads_week = df[(df['DATE_parsed'] >= week_start) & (df['DATE_parsed'] <= today)].shape[0]
month_start = today.replace(day=1)
leads_month = df[(df['DATE_parsed'] >= month_start) & (df['DATE_parsed'] <= today)].shape[0]

# Calculate previous week and month for comparison
prev_week_start = week_start - pd.Timedelta(days=7)
prev_week_end = week_start - pd.Timedelta(days=1)
leads_prev_week = df[(df['DATE_parsed'] >= prev_week_start) & (df['DATE_parsed'] <= prev_week_end)].shape[0]
prev_month = (month_start - pd.Timedelta(days=1)).replace(day=1)
prev_month_end = month_start - pd.Timedelta(days=1)
leads_prev_month = df[(df['DATE_parsed'] >= prev_month) & (df['DATE_parsed'] <= prev_month_end)].shape[0]
# Calculate deltas and arrows
week_delta = leads_week - leads_prev_week
month_delta = leads_month - leads_prev_month
week_arrow = f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if week_delta > 0 else (f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if week_delta < 0 else '')
month_arrow = f'<span style="color:#2ECC40;font-size:32px;vertical-align:middle;">&#9650;</span>' if month_delta > 0 else (f'<span style="color:#FF4136;font-size:32px;vertical-align:middle;">&#9660;</span>' if month_delta < 0 else '')
# Calculate percentage changes
if leads_prev_week > 0:
    week_pct = (week_delta / leads_prev_week) * 100
    week_pct_str = f'<span style="color:{"#2ECC40" if week_delta > 0 else "#FF4136"}; font-size:20px; font-weight:bold;">{abs(week_pct):.1f}%</span>'
else:
    week_pct_str = ''
if leads_prev_month > 0:
    month_pct = (month_delta / leads_prev_month) * 100
    month_pct_str = f'<span style="color:{"#2ECC40" if month_delta > 0 else "#FF4136"}; font-size:20px; font-weight:bold;">{abs(month_pct):.1f}%</span>'
else:
    month_pct_str = ''
# --- Cards ---
total_leads = len(df)

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

# Convert all columns to string to avoid ArrowTypeError
if not df.empty:
    df = df.astype(str)
# --- Table ---

# --- Counts ---
col_branch, col_source = st.columns([1.5, 1.2])
with col_branch:
    st.subheader("Leads per Branch and Source")
    if 'BRANCH' in df.columns and 'SOURCE OF LEAD GENERATION' in df.columns:
        # Create a pivot table: rows=BRANCH, columns=SOURCE OF LEAD GENERATION, values=count
        branch_source = pd.pivot_table(df, index='BRANCH', columns='SOURCE OF LEAD GENERATION', aggfunc='size', fill_value=0)
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
    st.subheader("Leads per Source")
    source_counts = df['SOURCE OF LEAD GENERATION'].value_counts()
    st.bar_chart(source_counts)

# Place Leads per Team Lead and Team Leader Conversion Ratios side by side
col_leads, col_conv = st.columns(2)
with col_leads:
    st.subheader("Leads per Team Lead")
    st.bar_chart(df['TEAM LEADER NAME'].value_counts())
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
        print('All TEAM LEADER NAME values before filtering:', sorted(df['TEAM LEADER NAME'].unique()))
        bad_names = ["undefined", "none", "null", "nan"]
        df_valid = df[df['TEAM LEADER NAME'].notna()].copy()
        df_valid['TLN_CLEAN'] = df_valid['TEAM LEADER NAME'].str.strip().str.lower()
        for bad in bad_names:
            df_valid = df_valid[~df_valid['TLN_CLEAN'].str.contains(bad)]
        df_valid = df_valid[df_valid['TLN_CLEAN'] != ""]
        print("Filtered unique team leader names:", sorted(df_valid['TEAM LEADER NAME'].unique()))
        print("First few rows of df_valid:")
        print(df_valid.head())
        df_valid['CONTACT_NORM'] = df_valid['CLIENT CONTACT'].apply(lambda x: normalize_contact(x)[0] if normalize_contact(x) else np.nan)
        total_leads = df_valid.groupby('TEAM LEADER NAME').size()
        df_valid['converted'] = df_valid['CONTACT_NORM'].apply(lambda x: x in sales_contacts if pd.notnull(x) else False)
        converted_leads = df_valid[df_valid['converted']].groupby('TEAM LEADER NAME').size()
        ratio = (converted_leads / total_leads).fillna(0).sort_values(ascending=False)
        mask = ratio.index.str.strip().str.lower() != 'undefined'
        print('Team leader names in ratio before final mask:', list(ratio.index))
        ratio = ratio[mask]
        print('Team leader names in ratio after final mask:', list(ratio.index))
        print("\n--- Team Leader Conversion Ratios (June & July Sales) ---")
        for tln, val in ratio.items():
            print(f"{tln}: {val:.4f}")
        print("--- End of Conversion Ratios ---\n")
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
            xaxis_title='Conversion Ratio (%)',
            yaxis_title='Team Leader',
            xaxis=dict(range=[0, max(15, conversion_perc.max() + 2)], tickformat='.0f'),
            margin=dict(l=20, r=20, t=40, b=40),
            width=650, height=350,
            showlegend=False,
            plot_bgcolor='white',
            title=None,
        )
        fig.update_xaxes(showgrid=True, gridcolor='#eee')
        fig.update_yaxes(autorange="reversed")
        st.markdown('<div style="overflow-x:auto;width=100%">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('CLIENT CONTACT or TEAM LEADER NAME column not found in the leads data.')

# After the cards and before the branch/source charts
st.markdown("---")

# --- Time Series Chart ---
st.subheader("Leads per Day - Time Series")
if 'DATE' in df.columns:
    # Clean and parse dates
    df_ts = df.copy()
    df_ts['DATE_parsed'] = pd.to_datetime(df_ts['DATE'], errors='coerce')
    
    # Remove invalid dates
    df_ts = df_ts.dropna(subset=['DATE_parsed'])
    
    # Count leads per day
    daily_leads = df_ts.groupby(df_ts['DATE_parsed'].dt.date).size().reset_index()
    daily_leads.columns = ['date', 'leads_count']
    daily_leads['date'] = pd.to_datetime(daily_leads['date'])
    
    # Sort by date
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
    
    # Update layout
    fig.update_layout(
        title=None,
        xaxis_title='Date',
        yaxis_title='Number of Leads',
        margin=dict(l=20, r=20, t=40, b=40),
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#eee',
        tickformat='%b %d, %Y'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#eee',
        zeroline=True,
        zerolinecolor='#ccc'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    st.info('No DATE column found in the data.')

st.markdown("---")

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Number of Agents per Branch")
    if 'BRANCH' in df.columns and 'AGENT NAME' in df.columns:
        branch_agent_counts = df.groupby('BRANCH')['AGENT NAME'].nunique().sort_values(ascending=False)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_bar(
            x=branch_agent_counts.values,
            y=branch_agent_counts.index,
            orientation='h',
            marker_color='#0074D9',
            hovertemplate='%{y}: %{x} agents<extra></extra>'
        )
        fig.update_layout(
            xaxis_title='Number of Agents',
            yaxis_title='Branch',
            margin=dict(l=20, r=20, t=30, b=40),
            width=650, height=350,
            showlegend=False,
            plot_bgcolor='white',
            title=None,
        )
        fig.update_xaxes(showgrid=True, gridcolor='#eee')
        fig.update_yaxes(autorange="reversed")
        st.markdown('<div style="overflow-x:auto;width:100%">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info('BRANCH or AGENT NAME column not found in the data.')
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