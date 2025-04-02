import os
import json
import bcrypt
from models.models import UserInDB
from logging import getLogger

logger = getLogger(__name__)


def load_users_from_json():
    users_file_path = 'users.json'
    try:
        if os.path.exists(users_file_path):
            with open(users_file_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Users file not found at {users_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading users from JSON: {str(e)}")
        return {}
    
def get_users_db():
    return load_users_from_json()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_user(db, username: str):
    # Load fresh user data each time to catch updates
    users_db = get_users_db()
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None