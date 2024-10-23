# Usa una imagen base de Python
FROM python:3.8-slim

# Copia los archivos de tu proyecto a la imagen
COPY . /app

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias necesarias
RUN pip install -r requirements.txt

# Expone el puerto en el que correrá Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]