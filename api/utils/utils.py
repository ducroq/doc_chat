from logging import getLogger
import re

logger = getLogger(__name__)

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

def validate_user_input_content(v: str) -> str:
    """
    Validate that user input does not contain malicious patterns.
    
    Args:
        v: The  string to validate
        
    Returns:
        str: The validated string
        
    Raises:
        ValueError: If the string contains dangerous patterns
    """
    # 1. Check for script injection patterns
    dangerous_patterns = [
        '<script>', 'javascript:', 'onload=', 'onerror=', 'onclick=',
        'ondblclick=', 'onmouseover=', 'onmouseout=', 'onfocus=', 'onblur=',
        'oninput=', 'onchange=', 'onsubmit=', 'onreset=', 'onselect=',
        'onkeydown=', 'onkeypress=', 'onkeyup=', 'ondragenter=', 'ondragleave=',
        'data:text/html', 'vbscript:', 'expression(', 'document.cookie',
        'document.write', 'window.location', 'eval(', 'exec('
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in v.lower():
            raise ValueError(f'Potentially unsafe input detected: {pattern}')
    
    # 2. Check for SQL injection patterns - Fixed regex
    sql_patterns = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION',
        'FROM', 'WHERE', '1=1', 'OR 1=1', 'OR TRUE', '--'
    ]
    
    # Count SQL keywords manually to avoid regex issues
    sql_count = 0
    for pattern in sql_patterns:
        # Check for whole words only
        if re.search(r'\b' + re.escape(pattern) + r'\b', v.upper()):
            sql_count += 1
    
    # Allow a few keywords as they might be in natural language
    if sql_count >= 3:
        raise ValueError('Potential SQL injection pattern detected')
    
    # 3. Check for command injection patterns
    cmd_patterns = [
        ';', '&&', '||', '`', '$(',  # Command chaining in bash/shell
        '| ', '>>', '>', '<', 'ping ', 'wget ', 'curl ', 
        'chmod ', 'rm -', 'sudo ', '/etc/', '/bin/'
    ]
    
    for pattern in cmd_patterns:
        if pattern in v:
            raise ValueError(f'Potential command injection pattern detected: {pattern}')
    
    # 4. Check for excessive special characters (might indicate an attack)
    special_char_count = sum(1 for char in v if char in '!@#$%^&*()+={}[]|\\:;"\'<>?/~`')
    if special_char_count > len(v) * 0.3:  # If more than 30% are special characters
        raise ValueError('Too many special characters in input')
        
    # 5. Check for extremely repetitive patterns (DoS attempts)
    if re.search(r'(.)\1{20,}', v):  # Same character repeated 20+ times
        raise ValueError('Input contains excessive repetition')
        
    return v

