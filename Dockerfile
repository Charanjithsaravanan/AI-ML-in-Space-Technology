FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GEOSAT_ALLOWED_ORIGINS=
COPY requirements.txt ./
COPY api/requirements.txt ./api-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api-requirements.txt
COPY . .
COPY --from=frontend /app/frontend/out ./frontend/out
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
