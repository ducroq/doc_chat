# Security Documentation

This document provides a comprehensive overview of the security features implemented in the EU-Compliant Document Chat system.

## Security Architecture

The system was designed with security in mind at every level:

1. **Perimeter Security**
   - Nginx reverse proxy with security headers
   - Rate limiting at multiple levels
   - Input validation and sanitization

2. **Authentication & Authorization**
   - Web interface authentication with bcrypt password hashing
   - API key-based authorization for API endpoints
   - Docker Secrets for credential management

3. **Network Security**
   - Isolation between frontend and backend networks
   - Internal components not exposed to public internet
   - Secure communication between components

4. **Application Security**
   - Comprehensive request validation
   - Protection against injection attacks (SQL, XSS, command)
   - Content filtering for uploaded documents

5. **Container Security**
   - Non-root user execution
   - Principle of least privilege
   - No-new-privileges restrictions

6. **Data Protection**
   - GDPR-compliant processing
   - Anonymization of personal identifiers
   - Automatic log rotation and deletion

## Implemented Security Controls

### Authentication

The web interface implements password-based authentication:
- Password hashing with bcrypt
- Session management
- Failed login attempt tracking

### Request Validation

All input is validated to prevent:
- Cross-site scripting (XSS)
- SQL injection
- Command injection
- Format string attacks
- Path traversal

### API Protection

API endpoints are protected with:
- API key validation
- Rate limiting by IP and overall requests
- Token budget enforcement
- Request validation
- Error handling that doesn't leak implementation details

### Secret Management

The system uses Docker Secrets for secure credential storage:
- Mistral API key
- Internal API key
- Automatic age checking for key rotation

### Rate Limiting

Multiple layers of rate limiting prevent abuse:
- IP-based rate limiting at the HTTP middleware level
- Global rate limiting for API requests
- Token budget limitations for external API calls

### Docker Security

Container security is enhanced through:
- Non-root user execution
- No-new-privileges flag
- Network isolation
- Volume mounting restrictions

### Data Protection

GDPR compliance is ensured through:
- Privacy-by-design principles
- Anonymization of user identifiers
- Configurable retention policies
- Explicit consent notifications

## Security Headers

The following security headers are configured in the Nginx reverse proxy:

```
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;
Referrer-Policy: strict-origin-when-cross-origin
```

## Authentication System

### Overview

The system implements a comprehensive authentication system that secures both the API and web interfaces using industry-standard security practices.

### Authentication Mechanisms

- **JWT-based Authentication**: JSON Web Tokens provide stateless authentication
- **Password Security**: Bcrypt hashing with appropriate work factors
- **Token Expiration**: Time-limited tokens with automatic expiration (30 minutes by default)
- **User Management**: Command-line tool for user administration
- **Role-Based Access**: Support for admin and regular user roles

### Implementation Details

#### Password Storage

Passwords are never stored in plain text. The system uses bcrypt with the following security features:

- Salted hashes unique to each user
- Configurable work factor (default: 12 rounds)
- Protection against timing attacks
- Automatic hashing of new passwords

#### Token Security

JWT tokens are secured with:

- HS256 encryption algorithm
- Secure random-generated signing key
- Short expiration timeframes
- Token invalidation on password change

#### User Management Security

The user management script enforces:

- Strong password requirements by default
- Proper permission controls
- Secure handling of generated passwords
- Immutable authentication logs

### Security Best Practices

To maintain authentication security:

1. **Rotate JWT Secret**: Periodically change the JWT signing secret
2. **Regular Password Changes**: Enforce password changes every 90 days
3. **Strong Passwords**: Use the `--generate-password` flag to create strong passwords
4. **Minimum Access**: Only grant admin access where necessary
5. **Account Deactivation**: Use `manage_users.py disable` rather than deleting accounts
6. **Audit User List**: Regularly review the active users with `manage_users.py list`

### Authentication Flows

#### API Authentication Flow

1. Client submits credentials to `/login` or `/token` endpoint
2. Server validates credentials against `users.json`
3. If valid, server generates a JWT token and returns it
4. Client includes token in `Authorization: Bearer <token>` header
5. Protected endpoints validate token via dependency injection

#### Web Interface Authentication Flow

1. User enters credentials in login form
2. Frontend sends credentials to API login endpoint
3. On success, token is stored in browser localStorage
4. Vue Router navigation guards protect routes
5. Automatic redirection to login page when token expires

### Common Attack Mitigations

- **Password Brute Force**: No rate limiting implemented yet (planned for future)
- **User Enumeration**: Generic error messages don't reveal valid usernames
- **Token Theft**: Short expiration time limits exposure if tokens are compromised
- **CSRF**: Tokens in Authorization header are immune to CSRF attacks

## Security Best Practices

When deploying and maintaining this system:

1. Regularly rotate API keys (system reminds after 90 days)
2. Keep all components updated
3. Monitor logs for suspicious activity
4. Use strong passwords for web interface
5. Place the system behind a firewall
6. Regularly perform security audits
7. Follow principle of least privilege for all accounts

## Security Recommendations

For even stronger security, consider:

1. Implementing multi-factor authentication
2. Adding an intrusion detection system
3. Regular penetration testing
4. Enabling Docker Content Trust for image verification
5. Implementing a Web Application Firewall (WAF)
6. Using TLS certificates for all communications
7. Implementing a more robust authentication system for production