import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample security log entries
security_logs = [
    "2023-10-01 08:30:15 Failed login attempt from IP 192.168.1.100",
    "2023-10-01 09:15:22 Successful login by user admin",
    "2023-10-01 10:45:30 Firewall blocked connection from IP 203.0.113.42",
    "2023-10-01 11:20:18 File deletion in /sensitive/data by user john",
    "2023-10-01 12:05:45 Multiple failed login attempts from IP 192.168.1.100",
    "2023-10-01 13:10:33 Unusual outbound traffic detected to IP 198.51.100.77",
    "2023-10-01 14:30:27 System update installed successfully",
    "2023-10-01 15:45:12 New user account created: alice",
    "2023-10-01 16:20:55 Suspicious file quarantined: malware.exe",
    "2023-10-01 17:05:40 Failed login attempt from IP 203.0.113.42"
]

def extract_ip_addresses(logs):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ip_addresses = [ip for log in logs for ip in re.findall(ip_pattern, log)]
    return ip_addresses

def extract_usernames(logs):
    username_pattern = r'user (\w+)'
    usernames = [match.group(1) for log in logs for match in re.finditer(username_pattern, log)]
    return usernames

def preprocess_logs(logs):
    stop_words = set(stopwords.words('english'))
    processed_logs = []
    for log in logs:
        tokens = word_tokenize(log.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        processed_logs.append(' '.join(tokens))
    return processed_logs

def identify_common_events(logs, top_n=5):
    words = [word for log in logs for word in log.split()]
    return Counter(words).most_common(top_n)

def cluster_logs(logs, n_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(logs)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_

def main():
    print("Extracting IP addresses:")
    ip_addresses = extract_ip_addresses(security_logs)
    print(ip_addresses)

    print("\nExtracting usernames:")
    usernames = extract_usernames(security_logs)
    print(usernames)

    processed_logs = preprocess_logs(security_logs)

    print("\nMost common events:")
    common_events = identify_common_events(processed_logs)
    for event, count in common_events:
        print(f"{event}: {count}")

    print("\nClustering logs:")
    clusters = cluster_logs(processed_logs)
    for i, (log, cluster) in enumerate(zip(security_logs, clusters)):
        print(f"Log {i + 1} - Cluster {cluster}: {log}")

    print("\nPotential security insights:")
    ip_counter = Counter(ip_addresses)
    for ip, count in ip_counter.items():
        if count > 1:
            print(f"Multiple events from IP {ip}")

    if 'failed login attempt' in ' '.join(security_logs).lower():
        print("Failed login attempts detected")

    if 'suspicious' in ' '.join(security_logs).lower():
        print("Suspicious activity detected")

if __name__ == "__main__":
    main()
