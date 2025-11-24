# Imagen base ligera
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Evitar archivos .pyc y asegurar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiar dependencias primero (optimiza cache)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Render asigna el puerto como variable PORT, pero exponemos 8050 por claridad
EXPOSE 8050

# Comando por defecto (se puede sobrescribir en Render)
CMD ["python", "dashboard.py"]
