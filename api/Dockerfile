FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Create an empty __init__.py if it doesn't exist
RUN touch __init__.py

# Create necessary directories and set permissions
RUN mkdir -p logs chat_data && \
    chmod -R 777 logs && \
    chmod -R 777 chat_data

# Run directly with Python instead of using module imports
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]