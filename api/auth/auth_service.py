import os
from logging import getLogger
import jwt
from jwt.exceptions import PyJWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

from config import settings
from models.models import User, TokenData
from auth.user_manager import get_user, verify_password 

logger = getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key")

async def authenticate_user(username: str, password: str):
    user = get_user(None, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(None, username=token_data.username)  # Remove the first parameter
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Validate API key for protected endpoints.
    
    Args:
        api_key: API key from request header
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    # Skip validation if no key file is configured
    if not os.path.exists(settings.INTERNAL_API_KEY_FILE):
        logger.warning(f"API key file not found: {settings.INTERNAL_API_KEY_FILE}")
        return api_key
    
    try:
        with open(settings.INTERNAL_API_KEY_FILE, "r") as f:
            expected_key = f.read().strip()
        
        if api_key != expected_key:
            raise HTTPException(
                status_code=403, 
                detail="Invalid API key"
            )
        
        return api_key
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error validating API key"
        )
    