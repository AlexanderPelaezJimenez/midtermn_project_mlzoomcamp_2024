from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

# Inicializar la aplicación Flask
app = Flask('bankruptcy')

model_file = 'model_C=1.0.bin'

# Cargar el modelo
def load_model():
    with open(model_file, 'rb') as f_in:
        model = pickle.load(f_in)
    return model

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos JSON de la solicitud
        data = request.get_json()
        
        # Convertir JSON a DataFrame
        df = pd.DataFrame([data])
        
        # Hacer predicción
        prediction = model.predict_proba(df)
        
        # Preparar respuesta
        response = {
            'probability_no_bankruptcy': float(prediction[0][0]),
            'probability_bankruptcy': float(prediction[0][1]),
            'prediction': 'Bankruptcy risk' if prediction[0][1] > 0.5 else 'No bankruptcy risk'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta de prueba para verificar que el servidor está funcionando
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)