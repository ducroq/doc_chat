import bcrypt

password = "password"
# Hash the password
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
print(hashed)