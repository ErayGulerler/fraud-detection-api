FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# using gunicorn for production, flask dev server for local
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api:app"]
