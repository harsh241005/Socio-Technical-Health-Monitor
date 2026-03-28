import pandas as pd

print("Loading Master Dataset...")
df = pd.read_csv("data/master_project_dataset.csv", low_memory=False)

# The Regex Filter we used in ISA III
bot_keywords = [
    'qbt Report', 'Jenkins', 'build failed', 'linux/x86_64', 'QA Test', 
    r'^\[jira\] \[Created\]', r'^\[jira\] \[Updated\]', r'^\[jira\] \[Commented\]',
    r'^\[jira\] \[Resolved\]', r'^\[jira\] \[Closed\]'
]
bot_pattern = '|'.join(bot_keywords)

# Create logic masks
is_flagged_by_regex = df['email_subject'].str.contains(bot_pattern, case=False, na=False, regex=True)
is_human_sender = ~df['sender'].str.contains('jira@|jenkins@|qa@', case=False, na=False)

print("\n=== AUDIT 1: Did our Regex accidentally delete HUMANS? ===")
false_positives = df[is_flagged_by_regex & is_human_sender]
print(f"Found {len(false_positives)} human emails flagged as bots by the regex.")
if len(false_positives) > 0:
    print("\nHere is what they look like (Check the sender and subject):")
    print(false_positives[['sender', 'email_subject']].head(10))

print("\n=== AUDIT 2: Did our Regex miss any BOTS? ===")
false_negatives = df[~is_flagged_by_regex & ~is_human_sender]
print(f"Found {len(false_negatives)} bot emails hiding in our 'human' dataset.")
if len(false_negatives) > 0:
    print("\nHere is what they look like:")
    print(false_negatives[['sender', 'email_subject']].head(10))