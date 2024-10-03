# (timestamp, event_type, severity, source_ip)
security_event = ('2023-09-30 14:30:00', 'Unauthorized Access Attempt', 'High', '203.0.113.42')

# Accessing event details
print(f"Event Type: {security_event[1]}")
print(f"Severity: {security_event[2]}")

# Trying to modify the tuple (this will raise an error)
# security_event[2] = 'Medium'  # This would raise a TypeError
