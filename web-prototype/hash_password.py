import bcrypt

password = "Y7i&90z209rn"
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
print(hashed)