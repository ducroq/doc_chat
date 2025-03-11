FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY static/ ./static/
COPY .streamlit/ ./.streamlit/


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]