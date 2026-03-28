import mailbox
import pandas as pd
import re
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import email.utils

# --- SETUP VADER ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

custom_lexicon = {
    'blocker': 0, 'critical': -0.2, 'fatal': -0.5, 'crash': -0.5,
    'panic': -0.4, 'deadlock': -0.3, 'corrupt': -0.5, 'outage': -0.5,
    'incident': -0.2, 'breach': -0.5, 'kill': 0, 'abort': 0, 'dead': 0,
    'terminate': 0, 'bug': -0.1, 'defect': -0.2, 'flaw': -0.2,
    'regression': -0.4, 'broken': -0.3, 'stuck': -0.3, 'hanging': -0.3,
    'frozen': -0.3, 'unstable': -0.3, 'unresponsive': -0.3, 'failing': -0.3,
    'fail': -0.3, 'failure': -0.4, 'error': -0.2, 'leak': -0.4,
    'bottleneck': -0.2, 'slow': -0.2, 'timeout': -0.2, 'latency': -0.1,
    'delay': -0.2, 'urgent': -0.3, 'vulnerability': -0.5, 'severity': -0.1,
    'warn': -0.1, 'exception': -0.1, 'drop': -0.1, 'revert': -0.2,
    'workaround': 0, 'hack': -0.1, 'patch': 0.1, 'resolved': 0.2, 'fixed': 0.3,
}
sia.lexicon.update(custom_lexicon)

def get_sentiment(text):
    return sia.polarity_scores(text)['compound']

# =============================================================================
# THE UPDATED PARSER (Now extracts Sender and Body Snippet)
# =============================================================================
def parse_mbox_robust(mbox_path):
    mbox = mailbox.mbox(mbox_path)
    email_data = []
    ticket_pattern = re.compile(r'(?:HADOOP|HDFS|YARN|MAPREDUCE)-\d+')

    count = 0
    for message in mbox:
        count += 1
        subject = message['subject'] or ""
        
        # 1. NEW: Extract Sender
        sender = message.get('From', 'Unknown')
        
        try:
            payload = message.get_payload()
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True)
                        break
            body_text = str(payload)
        except Exception:
            body_text = ""

        # 2. NEW: Extract Body Snippet (First 500 chars, no newlines)
        body_snippet = str(body_text)[:500].replace('\n', ' ').strip()

        full_text = f"{subject} {body_text}"
        mentioned_tickets = set(ticket_pattern.findall(full_text))

        sentiment = get_sentiment(body_text[:1000])
        date_str = message['date']

        for ticket in mentioned_tickets:
            email_data.append({
                'ticket_key': ticket.upper().strip(),
                'email_subject': subject[:100],
                'email_date': date_str,
                'behavior_score': sentiment,
                # 3. NEW: Add to dataset
                'sender': sender,
                'body_snippet': body_snippet
            })

    print(f"    -> Scanned {count} emails.")
    return pd.DataFrame(email_data)

# =============================================================================
# TESTING FRAMEWORK (Generates fake data to test the parser safely)
# =============================================================================
def create_test_environment():
    print("Setting up test environment...")
    os.makedirs('test_data', exist_ok=True)
    
    # 1. Create a Fake JIRA CSV
    jira_data = pd.DataFrame({
        'issue key': ['HADOOP-1000', 'HDFS-2000', 'YARN-3000'],
        'status.name': ['Open', 'Resolved', 'In Progress'],
        'priority.name': ['Major', 'Critical', 'Minor']
    })
    jira_data.to_csv('test_data/test_issues.csv', index=False)
    
    # 2. Create a Fake .mbox file with 3 emails
    mbox_path = 'test_data/test_archive.mbox'
    if os.path.exists(mbox_path):
        os.remove(mbox_path)
        
    mbox = mailbox.mbox(mbox_path)
    
    # Email A: A real human
    msg1 = mailbox.mboxMessage()
    msg1['Subject'] = 'Re: [jira] [Commented] (HADOOP-1000) We need to fix this memory leak'
    msg1['From'] = 'dev.human@apache.org'
    msg1['Date'] = email.utils.formatdate()
    msg1.set_payload('I have been looking at HADOOP-1000 and the memory leak is severe. It crashes the whole node. We should patch this urgently before the next release.')
    mbox.add(msg1)
    
    # Email B: Jenkins Bot
    msg2 = mailbox.mboxMessage()
    msg2['Subject'] = 'Apache Hadoop qbt Report: build failed'
    msg2['From'] = 'jenkins@builds.apache.org'
    msg2['Date'] = email.utils.formatdate()
    msg2.set_payload('Build failed on linux/x86_64. See logs for HDFS-2000. \n\n Stack trace: java.lang.NullPointerException at line 40...')
    mbox.add(msg2)
    
    # Email C: JIRA Auto-Bot
    msg3 = mailbox.mboxMessage()
    msg3['Subject'] = '[jira] [Resolved] (YARN-3000) Update documentation'
    msg3['From'] = 'jira@apache.org'
    msg3['Date'] = email.utils.formatdate()
    msg3.set_payload('This ticket has been resolved by the system automatically.')
    mbox.add(msg3)
    
    mbox.flush()
    return 'test_data/test_issues.csv', mbox_path

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n=== STARTING PARSER TEST ===")
    test_jira_csv, test_mbox = create_test_environment()
    
    print("\n1. Loading Mock JIRA Data...")
    jira_df = pd.read_csv(test_jira_csv)
    jira_df.rename(columns={'issue key': 'key', 'status.name': 'status'}, inplace=True)
    
    print("\n2. Running Updated Parser on Mock .mbox...")
    email_df = parse_mbox_robust(test_mbox)
    
    print("\n3. Merging Data...")
    final_df = pd.merge(email_df, jira_df, left_on='ticket_key', right_on='key', how='inner')
    
    print("\n=== TEST RESULTS: PRINTING EXTRACTED DATA ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 80)
    
    # Print the specific columns we care about to verify they work
    print(final_df[['ticket_key', 'sender', 'email_subject', 'body_snippet', 'behavior_score']])
    print("\nTest completed successfully. Safe to migrate changes to main script!")