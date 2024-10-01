suspicious_ips = ['192.168.1.100', '10.0.0.25', '172.16.0.1']

# Add a new suspicious IP
suspicious_ips.append('192.168.0.50')

# Remove an IP that's no longer suspicious
suspicious_ips.remove('10.0.0.25')

# Check if an IP is in the suspicious list
ip_to_check = '172.16.0.1'
if ip_to_check in suspicious_ips:
    print(f"{ip_to_check} is suspicious!")

print(suspicious_ips)
