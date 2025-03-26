# Authentication System

The EU-Compliant Document Chat system implements a token-based authentication system to secure both the API and web interfaces. This document explains the authentication architecture, user management, and how to work with the system.

## Authentication Architecture

The system uses a JWT (JSON Web Token) based authentication flow:

1. **User Login**: Users provide username and password to authenticate
2. **Token Generation**: The server validates credentials and issues a JWT token
3. **Authenticated Requests**: The token is included in subsequent requests 
4. **Authorization**: The server validates the token for protected endpoints

### Components

- **User Storage**: Users and credentials are stored in `users.json`
- **Password Security**: Passwords are stored as bcrypt hashes
- **Token Management**: JWT tokens with configurable expiration time
- **Management Tool**: Command-line user management script

## User Management

### Using the Management Script

The `manage_users.py` script provides a command-line interface for managing users:

```bash
# Create a new user
python manage_users.py create username password --full-name "User Name" --email "user@example.com"

# Create an admin user
python manage_users.py create username password --admin

# Generate a secure password
python manage_users.py create username --generate-password

# List all users
python manage_users.py list

# Enable/disable a user
python manage_users.py disable username
python manage_users.py enable username

# Reset a user's password
python manage_users.py reset-password username newpassword
python manage_users.py reset-password username --generate
```

### User Configuration

Users are stored in the `users.json` file with the following structure:

```json
{
  "admin": {
    "username": "admin",
    "full_name": "Administrator",
    "email": "admin@example.com",
    "hashed_password": "$2b$12$...",
    "disabled": false,
    "is_admin": true
  }
}
```

This file is automatically loaded by the API service and controls who can access the system.

## Security Considerations

- JWT tokens expire after 30 minutes by default (configurable)
- Passwords are hashed using bcrypt with appropriate work factors
- The system enforces strong password requirements (can be bypassed with `--force`)
- Failed login attempts are logged
- Authentication failures return generic messages to prevent enumeration attacks

## Implementation Details

### API Authentication

The API uses FastAPI's dependency injection system to protect endpoints:

```python
# Protected endpoint example
@app.get("/protected-route")
async def protected_route(current_user: User = Depends(get_current_active_user)):
    return {"message": "This is protected", "user": current_user.username}
```

### Frontend Authentication

The frontend stores the JWT token in localStorage and includes it in all API requests. When the token expires, the user is redirected to the login page.

## Configuration

Authentication-related configuration is controlled by environment variables:

- `JWT_SECRET_KEY_FILE`: Path to the file containing the JWT secret key
- `JWT_ALGORITHM`: Algorithm used for JWT (default: HS256)
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: Token validity period (default: 30)

## Troubleshooting

### Common Issues

1. **Failed Login**:
   - Verify username and password
   - Check if user is disabled
   - Ensure users.json is properly mounted in the container

2. **Token Expiration**:
   - Default token validity is 30 minutes
   - Extend by changing JWT_ACCESS_TOKEN_EXPIRE_MINUTES

3. **User Management**:
   - The API must be restarted to pick up user.json changes
   - Use docker-compose restart api after major user changes