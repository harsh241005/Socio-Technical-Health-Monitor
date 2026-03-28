import pandas as pd

print("=== 1. CHECKING TARGET LEAKAGE ===")
# Load the currently enriched dataset
df_enriched = pd.read_csv("data/processed/isa3_enriched_dataset.csv")

print("\n--- is_assigned vs is_stalled ---")
# This will show if unassigned tickets are overwhelmingly stalled
print(pd.crosstab(df_enriched['is_stalled'], df_enriched['is_assigned'], normalize='index') * 100)

print("\n--- has_votes vs is_stalled ---")
print(pd.crosstab(df_enriched['is_stalled'], df_enriched['has_votes'], normalize='index') * 100)


print("\n=== 2. CHECKING BOT FILTER OVER-CORRECTION ===")
# Load the RAW dataset to see what we accidentally dropped
df_raw = pd.read_csv("data/master_project_dataset.csv", low_memory=False)

bot_keywords = ['qbt Report', 'Jenkins', 'build failed', 'linux/x86_64', 'QA Test', r'\[jira\]']
bot_pattern = '|'.join(bot_keywords)

# Find exactly which rows got flagged as bots
dropped_rows = df_raw[df_raw['email_subject'].str.contains(bot_pattern, case=False, na=False, regex=True)]

print(f"\nTotal rows caught by filter: {len(dropped_rows)}")
print("\nTop 15 Subject Lines Dropped (Look for 'Re: [jira]' - these are human!):")
print(dropped_rows['email_subject'].value_counts().head(15))