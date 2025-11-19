FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app ./app
COPY src ./src
COPY model ./model

EXPOSE 5000

CMD ["python", "app/app.py"]
