# Save this as manage_users.py in the project root directory
import bcrypt
import json
import os
import argparse
from api.utils import validate_password

USER_DB_FILE = "users.json"

def load_users():
    """Load users from the JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to the JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user(username, password, full_name=None, email=None, is_admin=False, force=False):
    """Create a new user or update an existing one"""
    # Validate password unless force is True
    if not force:
        is_valid, message = validate_password(password)
        if not is_valid:
            print(f"Password validation failed: {message}")
            print("Use --force to override password validation")
            return False
    
    users = load_users()
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Create or update user entry
    users[username] = {
        "username": username,
        "full_name": full_name or username,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False,
        "is_admin": is_admin
    }
    
    save_users(users)
    print(f"User '{username}' {'updated' if username in users else 'created'} successfully.")
    return True

def disable_user(username, disable=True):
    """Disable or enable a user"""
    users = load_users()
    
    if username not in users:
        print(f"User '{username}' does not exist!")
        return
    
    users[username]["disabled"] = disable
    save_users(users)
    
    status = "disabled" if disable else "enabled"
    print(f"User '{username}' {status} successfully.")

def delete_user(username):
    """Delete a user"""
    users = load_users()
    
    if username not in users:
        print(f"User '{username}' does not exist!")
        return
    
    del users[username]
    save_users(users)
    print(f"User '{username}' deleted successfully.")

def list_users():
    """List all users"""
    users = load_users()
    
    if not users:
        print("No users found.")
        return
    
    print("\nUser List:")
    print("=" * 60)
    print(f"{'Username':<15} {'Full Name':<20} {'Email':<20} {'Status':<10}")
    print("-" * 60)
    
    for username, user in users.items():
        status = "Disabled" if user.get("disabled", False) else "Active"
        if user.get("is_admin", False):
            status += " (Admin)"
        
        print(f"{username:<15} {user.get('full_name', ''):<20} {user.get('email', ''):<20} {status:<10}")
    
    print("=" * 60)

def generate_password():
    """Generate a strong password"""
    import random
    import string
    
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = '@#$%^&+=!'
    
    # Ensure at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    # Add remaining characters
    all_chars = lowercase + uppercase + digits + special
    password.extend(random.choice(all_chars) for _ in range(8))
    
    # Shuffle the password
    random.shuffle(password)
    
    # Convert to string
    return ''.join(password)

def main():
    parser = argparse.ArgumentParser(description="User management for EU-Compliant Document Chat")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Create user command
    create_parser = subparsers.add_parser("create", help="Create or update a user")
    create_parser.add_argument("username", help="Username")
    create_parser.add_argument("password", help="Password", nargs="?")
    create_parser.add_argument("--full-name", help="Full name")
    create_parser.add_argument("--email", help="Email address")
    create_parser.add_argument("--admin", action="store_true", help="Set as admin user")
    create_parser.add_argument("--force", action="store_true", help="Force creation even if password doesn't meet requirements")
    create_parser.add_argument("--generate-password", action="store_true", help="Generate a strong password")
    
    # List users command
    list_parser = subparsers.add_parser("list", help="List all users")
    
    # Disable user command
    disable_parser = subparsers.add_parser("disable", help="Disable a user")
    disable_parser.add_argument("username", help="Username")
    
    # Enable user command
    enable_parser = subparsers.add_parser("enable", help="Enable a user")
    enable_parser.add_argument("username", help="Username")
    
    # Delete user command
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("username", help="Username")
    
    # Reset password command
    reset_parser = subparsers.add_parser("reset-password", help="Reset a user's password")
    reset_parser.add_argument("username", help="Username")
    reset_parser.add_argument("password", help="New password", nargs="?")
    reset_parser.add_argument("--generate", action="store_true", help="Generate a strong password")
    reset_parser.add_argument("--force", action="store_true", help="Force reset even if password doesn't meet requirements")
    
    args = parser.parse_args()
    
    if args.command == "create":
        password = args.password
        if args.generate_password or (not password and args.password is None):
            password = generate_password()
            print(f"Generated password: {password}")
        elif not password:
            import getpass
            password = getpass.getpass("Enter password: ")
            confirmation = getpass.getpass("Confirm password: ")
            if password != confirmation:
                print("Passwords do not match!")
                return
        
        create_user(args.username, password, args.full_name, args.email, args.admin, args.force)
    
    elif args.command == "list":
        list_users()
    
    elif args.command == "disable":
        disable_user(args.username, True)
    
    elif args.command == "enable":
        disable_user(args.username, False)
    
    elif args.command == "delete":
        delete_user(args.username)
    
    elif args.command == "reset-password":
        users = load_users()
        if args.username not in users:
            print(f"User '{args.username}' does not exist!")
            return
        
        password = args.password
        if args.generate or (not password and args.password is None):
            password = generate_password()
            print(f"Generated password: {password}")
        elif not password:
            import getpass
            password = getpass.getpass("Enter new password: ")
            confirmation = getpass.getpass("Confirm new password: ")
            if password != confirmation:
                print("Passwords do not match!")
                return
        
        is_admin = users[args.username].get("is_admin", False)
        full_name = users[args.username].get("full_name")
        email = users[args.username].get("email")
        
        create_user(args.username, password, full_name, email, is_admin, args.force)
        print(f"Password reset for user '{args.username}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()