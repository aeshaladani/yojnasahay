# Use Python 3.11 slim image (matches  on Render)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]