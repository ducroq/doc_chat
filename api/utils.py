import re
import random
import hashlib
import time

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

def generate_math_captcha():
    """Generate a simple math problem as CAPTCHA"""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
        question = f"{a} + {b}"
    elif operation == '-':
        # Ensure positive result
        if b > a:
            a, b = b, a
        answer = a - b
        question = f"{a} - {b}"
    else:  # multiplication
        answer = a * b
        question = f"{a} × {b}"
    
    # Create a hash of the answer with a time-based salt
    timestamp = int(time.time())
    answer_hash = hashlib.sha256(f"{answer}:{timestamp}".encode()).hexdigest()
    
    return {
        "question": question,
        "hash": answer_hash,
        "timestamp": timestamp
    }

def verify_math_captcha(user_answer, answer_hash, timestamp):
    """Verify the math CAPTCHA answer"""
    # Ensure timestamp is an integer
    try:
        timestamp = int(timestamp)
        current_time = int(time.time())
        
        # Check if CAPTCHA has expired (10 minutes)
        if current_time - timestamp > 600:  # This is where the error was happening
            return False
        
        # Convert user_answer to int and verify
        user_answer = int(user_answer)
        check_hash = hashlib.sha256(f"{user_answer}:{timestamp}".encode()).hexdigest()
        return check_hash == answer_hash
    except (ValueError, TypeError) as e:
        # Log the error for debugging
        print(f"CAPTCHA verification error: {e}")
        return False