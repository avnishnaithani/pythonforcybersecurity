failed_logins = {'alice', 'bob', 'charlie', 'alice', 'david', 'bob'}

# The set automatically removes duplicates
print(failed_logins)  # Output: {'alice', 'bob', 'charlie', 'david'}

# Quick check if a username has had failed attempts
username_to_check = 'eve'
if username_to_check in failed_logins:
    print(f"{username_to_check} has had failed login attempts")
else:
    print(f"No failed attempts for {username_to_check}")

# Add a new failed attempt
failed_logins.add('eve')

# Set operations
allowed_users = {'alice', 'charlie', 'eve', 'frank'}
suspicious_users = failed_logins - allowed_users
print(f"Suspicious users: {suspicious_users}")
