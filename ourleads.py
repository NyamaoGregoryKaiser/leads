import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
from datetime import datetime

# --- Instructions ---
# 1. Go to https://console.developers.google.com/
# 2. Create a project and enable the Google Sheets API.
# 3. Create a service account and download the JSON credentials file.
# 4. Share your Google Sheet with the service account email (from the JSON file).
# 5. Save the credentials file as 'credentials.json' in the same directory as this script.

# --- Google Sheets Setup ---
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1CNR-5kLURpyEjbKVqA_ejtzgSdDsXMD9EmWtO7t2Ml0/edit?usp=sharing'
SHEET_ID = SHEET_URL.split('/')[5]

# Define the scope
scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]

# Authenticate and connect
def get_leads_dataframe():
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.sheet1  # Assumes data is in the first sheet
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    return df

def main():
    df = get_leads_dataframe()
    # --- Visualisation Cards: Today, This Week, This Month ---
    today = pd.Timestamp(datetime.now().date())
    if 'DATE' in df.columns:
        df['DATE_parsed'] = pd.to_datetime(df['DATE'], errors='coerce')
        leads_today = df[df['DATE_parsed'].dt.date == today.date()].shape[0]
        # This week (Monday to today)
        week_start = today - pd.Timedelta(days=today.weekday())
        leads_week = df[(df['DATE_parsed'] >= week_start) & (df['DATE_parsed'] <= today)].shape[0]
        # This month
        month_start = today.replace(day=1)
        leads_month = df[(df['DATE_parsed'] >= month_start) & (df['DATE_parsed'] <= today)].shape[0]
    else:
        leads_today = 0
        leads_week = 0
        leads_month = 0
    # Create cards in a row
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
    card_data = [
        (leads_today, 'Leads Today', '#00FFD0'),
        (leads_week, 'Leads This Week', '#FFA500'),
        (leads_month, 'Leads This Month', '#00FF66'),
    ]
    for ax, (val, label, color) in zip(axes, card_data):
        fig.patch.set_facecolor('#181818')
        ax.set_facecolor('#181818')
        ax.text(0.5, 0.6, f"{val}", fontsize=48, color=color, ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.18, label, fontsize=18, color='white', ha='center', va='center', fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    # --- End Cards ---
    print('Sample of loaded leads:')
    print(df.head())
    print('\nLeads per Branch:')
    print(df['BRANCH'].value_counts())
    print('\nLeads per Agent:')
    print(df['AGENT NAME'].value_counts())
    print('\nLeads per Source of Lead Generation:')
    print(df['SOURCE OF LEAD GENERATION'].value_counts())

if __name__ == '__main__':
    main() 