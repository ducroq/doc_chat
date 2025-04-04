FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs && \
    chown -R 777 logs

# Run the application
CMD ["python", "main.py"]