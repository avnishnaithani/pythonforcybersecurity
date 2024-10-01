user_access_levels = {
    'alice': 'admin',
    'bob': 'user',
    'charlie': 'moderator',
    'david': 'user'
}

# Check a user's access level
user_to_check = 'bob'
if user_to_check in user_access_levels:
    print(f"{user_to_check}'s access level: {user_access_levels[user_to_check]}")
else:
    print(f"No access level found for {user_to_check}")

# Update a user's access level
user_access_levels['charlie'] = 'admin'

# Add a new user
user_access_levels['eve'] = 'user'

# Iterate through all users and their access levels
for username, access_level in user_access_levels.items():
    print(f"User: {username}, Access Level: {access_level}")
