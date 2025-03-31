import re

def validate_password(password):
    """
    Validate password against requirements:
    - At least 8 characters
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one of these special characters: @#$%^&+=!
    - Doesn't contain problematic characters like quotes, backticks, or backslashes
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[@#$%^&+=!]', password):
        return False, "Password must contain at least one special character (@#$%^&+=!)"
    
    # Check for problematic characters
    if re.search(r'[\'"`\\]', password):
        return False, "Password contains invalid characters (quotes, backticks, or backslashes are not allowed)"
    
    # If all checks pass
    return True, "Password is valid"
