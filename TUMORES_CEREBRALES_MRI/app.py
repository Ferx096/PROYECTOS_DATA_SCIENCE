from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
modelo = load_model("modelo_mri_tumor.h5")

def preproceso_imagenes(img_path):
    imagen = image.load_img(img_path, target_size=(224,224))
    
    # Convertir la imagen a array de numpy
    img_array = image.img_to_array(imagen)/255.0  # Escalar la imagen a valores entre 0 y 1
    
    # Expandir dimensiones para que coincida con el formato de entrada del modelo (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Crear la carpeta 'uploads' si no existe
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Intentar preprocesar la imagen
        try:
            img_array = preproceso_imagenes(file_path)
            prediction = modelo.predict(img_array)
        except Exception as e:
            return jsonify({"error": str(e)}), 500  # Devolver el error si algo falla
        
        class_names = ['glioma','meningioma','notumor', 'pituitary']
        predicted_class = class_names[np.argmax(prediction)]
        
        return jsonify({"prediction": predicted_class})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
