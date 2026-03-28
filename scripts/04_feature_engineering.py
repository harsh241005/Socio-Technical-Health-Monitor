"""
=============================================================================
SOCIO-TECHNICAL HEALTH MONITOR
Feature Engineering Pipeline — ISA III
=============================================================================

WHAT THIS SCRIPT DOES:
  Reads the raw master_project_dataset.csv and produces a fully enriched
  dataset with all features needed for ISA III model training.

CHANGES FROM ISA II:
  - Fixed is_stalled definition (old new-code definition was wrong — see notes)
  - Added 14 new features across 5 categories
  - Dropped 100%-null customfield_12310921
  - Added defensive handling for MIME-encoded subjects
  - Added has_enough_emails flag for sentiment_trend reliability
  - [NEW] Implemented Silver Bullet "Sender-based" Bot Filter
  - [NEW] Isolated "Honest" signals from target-leaking proxy variables

FEATURES PRODUCED (28 total engineered, ~7 "Honest" used for ML):
  Category 1 — Time-based       : days_to_resolve, days_since_last_update
  Category 2 — Email signals    : subject_length, email_count_per_ticket
                                  has_enough_emails
  Category 3 — Engagement       : votes.votes, has_votes, watches.watchcount
                                  is_assigned
  Category 4 — Sentiment        : behavior_score (raw), avg_sentiment,
                                  sentiment_variance, sentiment_trend
  Category 5 — Categorical      : priority_numeric, issuetype one-hot (7 cols),
                                  project one-hot (4 cols)
  Target variable               : is_stalled
=============================================================================
"""

import pandas as pd
import numpy as np
import re
from email.header import decode_header

# ---- CONFIGURATION ----
INPUT_FILE  = "data/interim/master_project_dataset.csv"
OUTPUT_FILE = "data/processed/isa3_enriched_dataset.csv"

# Statuses that mean a ticket is resolved/done
RESOLVED_STATUSES = {'Resolved', 'Closed'}

# Priority ordering (Trivial = least urgent, Blocker = most urgent)
PRIORITY_MAP = {
    'Trivial' : 1,
    'Minor'   : 2,
    'Major'   : 3,
    'Critical': 4,
    'Blocker' : 5,
}

# ============================================================================
# STEP 0 — LOAD
# ============================================================================
print("=" * 60)
print("SOCIO-TECHNICAL HEALTH MONITOR — Feature Engineering ISA III")
print("=" * 60)

print("\n[Step 0] Loading master dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Loaded: {len(df)} rows, {df.shape[1]} columns")


# ============================================================================
# STEP 1 — DATE PARSING & CLEANING
# ============================================================================
print("\n[Step 1] Parsing dates...")

df['created_dt']        = pd.to_datetime(df['created'],        errors='coerce', utc=True)
df['resolutiondate_dt'] = pd.to_datetime(df['resolutiondate'], errors='coerce', utc=True)
df['email_dt']          = pd.to_datetime(df['email_date'],     errors='coerce', utc=True)
df['updated_dt']        = pd.to_datetime(df['updated'],        errors='coerce', utc=True)

before = len(df)
df = df.dropna(subset=['created_dt', 'email_dt'])
dropped = before - len(df)
print(f"  Dropped {dropped} rows with unparseable dates")
print(f"  Rows remaining: {len(df)}")


# ============================================================================
# STEP 2 — SUBJECT LINE CLEANING (MIME decoding)
# ============================================================================
print("\n[Step 2] Cleaning email subjects (MIME decoding)...")

def decode_mime_subject(subject):
    """
    Converts MIME-encoded subjects like:
      =?utf-8?Q?[jira]_[Created]_(HADOOP-18706)?=
    into readable text:
      [jira] [Created] (HADOOP-18706) The temporary files...
    """
    try:
        parts = decode_header(str(subject))
        decoded = ''
        for part, enc in parts:
            if isinstance(part, bytes):
                decoded += part.decode(enc or 'utf-8', errors='replace')
            else:
                decoded += str(part)
        # Remove any residual encoding artifacts
        decoded = re.sub(r'=\?utf-8\?[qQbB]\?[^?]*\?=', '', decoded)
        return decoded.strip()
    except Exception:
        return subject

df['email_subject_clean'] = df['email_subject'].apply(decode_mime_subject)
encoded_count = df['email_subject'].str.contains('utf-8', na=False).sum()
print(f"  Decoded {encoded_count} MIME-encoded subjects")


# ============================================================================
# STEP 3 — DROP USELESS COLUMNS
# ============================================================================
print("\n[Step 3] Dropping useless columns...")

cols_to_drop = [
    'customfield_12310921',   # 100% null — confirmed in audit
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
print(f"  Dropped: {cols_to_drop}")


# ============================================================================
# STEP 4 — CATEGORY 1: TIME-BASED FEATURES
# ============================================================================
print("\n[Step 4] Engineering time-based features...")

# Feature: days_to_resolve
# NOTE: do not use this as a model feature directly — it leaks the target.
df['days_to_resolve'] = (df['resolutiondate_dt'] - df['created_dt']).dt.days
print(f"  days_to_resolve: mean={df['days_to_resolve'].mean():.0f} days "
      f"(nulls={df['days_to_resolve'].isna().sum()} = stalled tickets)")

# Feature: days_since_last_update
df['days_since_last_update'] = (
    (df['updated_dt'] - df['created_dt']).dt.days.fillna(0)
)
print(f"  days_since_last_update: mean={df['days_since_last_update'].mean():.0f} days")


# ============================================================================
# STEP 5 — TARGET VARIABLE: is_stalled
# ============================================================================
print("\n[Step 5] Creating target variable: is_stalled...")

# CORRECT DEFINITION: a ticket is stalled if it has NO resolution date.
df['is_stalled'] = np.where(df['resolutiondate_dt'].isna(), 1, 0)

stalled_n   = df['is_stalled'].sum()
stalled_pct = df['is_stalled'].mean() * 100
print(f"  Stalled tickets : {stalled_n} ({stalled_pct:.1f}%)")
print(f"  Active tickets  : {len(df) - stalled_n} ({100 - stalled_pct:.1f}%)")
print(f"  Class ratio     : {(len(df)-stalled_n)/stalled_n:.1f}:1 (active:stalled)")
print(f"  NOTE: Use class_weight='balanced' in model to handle imbalance")


# ============================================================================
# STEP 6 — EMAIL SIGNAL FEATURES & BOT ERADICATION
# ============================================================================
print("\n[Step 6] Engineering email signals & filtering bots...")

df['subject_length'] = (
    df['email_subject_clean'].astype(str).apply(lambda x: len(x.split()))
)

# --- THE SILVER BULLET BOT FILTER ---
# Instead of brittle regex, we use the absolute ground truth: the sender address.
original_len = len(df)

if 'sender' not in df.columns:
    print("  [WARNING] 'sender' column not found! Ensure entity_linking_parser.py was run correctly.")
    df['sender'] = ''

bot_senders = 'jira@|jenkins@|qa@'
df = df[~df['sender'].str.contains(bot_senders, case=False, na=False)]

bots_dropped = original_len - len(df)
print(f"  *** DROPPED {bots_dropped} bot-generated emails (Sender-Verified) ***")

# Feature: email_count_per_ticket (Now safely counting ONLY human emails)
email_counts = (
    df.groupby('ticket_key')['email_subject']
      .count()
      .reset_index()
)
email_counts.columns = ['ticket_key', 'email_count_per_ticket']
df = df.merge(email_counts, on='ticket_key', how='inner')

# Feature: has_enough_emails (For sentiment_trend reliability)
df['has_enough_emails'] = (df['email_count_per_ticket'] >= 2).astype(int)

print(f"  subject_length          : mean={df['subject_length'].mean():.1f} words")
print(f"  email_count_per_ticket  : mean={df['email_count_per_ticket'].mean():.1f} (Human only)")
print(f"  has_enough_emails       : {df['has_enough_emails'].sum()} rows with >=2 emails")

# ============================================================================
# STEP 7 — CATEGORY 3: ENGAGEMENT FEATURES
# ============================================================================
print("\n[Step 7] Engineering engagement features...")

df['priority_numeric'] = df['priority'].map(PRIORITY_MAP).fillna(0)
df['votes.votes'] = df['votes.votes'].fillna(0)
df['has_votes'] = (df['votes.votes'] > 0).astype(int)
df['watches.watchcount'] = df['watches.watchcount'].fillna(0)

# Feature: is_assigned
# Target Leakage Note: Unassigned tickets heavily overlap with stalled tickets.
# This will be excluded from the final 'Honest' model evaluation.
df['is_assigned'] = df['assignee'].notna().astype(int)

print(f"  priority_numeric    : {df['priority_numeric'].value_counts().to_dict()}")
print(f"  has_votes           : {df['has_votes'].sum()} tickets with votes")
print(f"  is_assigned         : {df['is_assigned'].sum()} assigned, "
      f"{(~df['assignee'].notna()).sum()} unassigned")


# ============================================================================
# STEP 8 — CATEGORY 4: SENTIMENT FEATURES
# ============================================================================
print("\n[Step 8] Engineering sentiment features...")

sentiment_stats = (
    df.groupby('ticket_key')['behavior_score']
      .agg(['mean', 'std'])
      .reset_index()
)
sentiment_stats.columns = ['ticket_key', 'avg_sentiment', 'sentiment_variance']
df = df.merge(sentiment_stats, on='ticket_key', how='left')

df['sentiment_variance'] = df['sentiment_variance'].fillna(0)

def compute_sentiment_trend(group):
    group = group.sort_values('email_date')
    mid = len(group) // 2
    if mid == 0:
        return 0.0  # Not enough emails to compute trend
    early_avg = group.iloc[:mid]['behavior_score'].mean()
    late_avg  = group.iloc[mid:]['behavior_score'].mean()
    return float(late_avg - early_avg)

# Fixed Deprecation Warning with include_groups=False
trend_df = (
    df.groupby('ticket_key', group_keys=False)
      .apply(lambda g: pd.Series({'sentiment_trend': compute_sentiment_trend(g)}), include_groups=False)
      .reset_index()
)
df = df.merge(trend_df, on='ticket_key', how='left')

print(f"  avg_sentiment       : mean={df['avg_sentiment'].mean():.3f}")
print(f"  sentiment_variance  : mean={df['sentiment_variance'].mean():.3f} "
      f"(stalled={df[df['is_stalled']==1]['sentiment_variance'].mean():.3f}, "
      f"active={df[df['is_stalled']==0]['sentiment_variance'].mean():.3f})")
print(f"  sentiment_trend     : mean={df['sentiment_trend'].mean():.3f}")


# ============================================================================
# STEP 9 — CATEGORY 5: CATEGORICAL ENCODING
# ============================================================================
print("\n[Step 9] Encoding categorical features...")

issuetype_dummies = pd.get_dummies(
    df['issuetype.name'], prefix='type', dtype=int
)
df = pd.concat([df, issuetype_dummies], axis=1)
print(f"  issuetype columns   : {list(issuetype_dummies.columns)}")

project_dummies = pd.get_dummies(
    df['project.key'], prefix='proj', dtype=int
)
df = pd.concat([df, project_dummies], axis=1)
print(f"  project columns     : {list(project_dummies.columns)}")


# ============================================================================
# STEP 10 — SUMMARY & SAVE
# ============================================================================
print("\n[Step 10] Final summary...")

# The 7 Pure Communication & Sentiment Signals (No Leakage)
HONEST_ML_FEATURES = [
    'email_count_per_ticket', 
    'subject_length',
    'avg_sentiment', 
    'sentiment_variance', 
    'sentiment_trend',
    'priority_numeric'
]
# Features identified as Target Leakage (Structural Proxies)
LEAKING_FEATURES = [
    'is_assigned', 
    'has_votes', 
    'votes.votes', 
    'watches.watchcount',
    'days_to_resolve'
]

print(f"\n  Total rows          : {len(df)}")
print(f"  Total columns       : {df.shape[1]}")
print(f"  Honest ML features  : {len(HONEST_ML_FEATURES)} pure signals")
print(f"  Stalled tickets     : {df['is_stalled'].sum()} ({df['is_stalled'].mean()*100:.1f}%)")
print(f"  Active tickets      : {(df['is_stalled']==0).sum()}")

print("\n  HONEST Feature correlations with is_stalled (Pure Signals Only):")
corr_data = []
for f in HONEST_ML_FEATURES:
    if f in df.columns:
        c = df[f].corr(df['is_stalled'])
        corr_data.append((f, c))
        
for feat, corr in sorted(corr_data, key=lambda x: abs(x[1]), reverse=True):
    bar = '█' * int(abs(corr) * 20)
    direction = '+' if corr > 0 else '-'
    print(f"    {feat:30s}: {corr:+.4f}  {direction}{bar}")

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved to: {OUTPUT_FILE}")
print("\n" + "=" * 60)
print("Feature engineering complete. Ready for 'Honest' model training.")
print("=" * 60)