from flask import Flask, request, jsonify
import librosa as lr
import numpy as np
import tensorflow as tf
import io
import os

app = Flask(__name__)

# labels = ['tat_quat', 'mo_cua', 'dong_cua', 'tat_den', 'bat_quat', 'bat_den']
CLASS_MAPPING = {
    0: "tat_quat",
    1: "mo_cua",
    2: "dong_cua",
    3: "tat_den",
    4: "bat_quat",
    5: "bat_den"
}
# checkpoint_filepath = './best_model_ver3.keras'
checkpoint_filepath = './best_model_ver6.keras'
# checkpoint_filepath = '/content/drive/MyDrive/Colab Notebooks/IoT/checkpoints/best_model_ver6.keras'
model = tf.keras.models.load_model(checkpoint_filepath)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# try:
#     # First, verify the file exists
#     if not os.path.exists(checkpoint_filepath):
#         raise FileNotFoundError(f"Model file not found at: {checkpoint_filepath}")
    
#     # Try loading the model
#     model = tf.keras.models.load_model(checkpoint_filepath)
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     # You might want to exit here or handle the error appropriately
#     raise

def augment_audio(data, sample_rate):
    augmented_data = [data]
    time_stretch = lr.effects.time_stretch(data, rate=0.9)
    augmented_data.append(time_stretch)
    pitch_shift_up = lr.effects.pitch_shift(data, sr=sample_rate, n_steps=2)
    pitch_shift_down = lr.effects.pitch_shift(data, sr=sample_rate, n_steps=-2)
    augmented_data.extend([pitch_shift_up, pitch_shift_down])
    noise = 0.005 * np.random.randn(len(data))
    noisy_data = data + noise
    augmented_data.append(noisy_data)
    amplitude_mod = data * 0.8
    augmented_data.append(amplitude_mod)
    return augmented_data

# def preprocess_audio(audio_data, sample_rate):
#     augmented_data = augment_audio(audio_data, sample_rate)
#     features = []
#     for augmented in augmented_data:
#         mfccs = np.mean(lr.feature.mfcc(y=augmented, sr=sample_rate).T, axis=0)
#         mel_spec = np.mean(lr.feature.melspectrogram(y=augmented, sr=sample_rate).T, axis=0)
#         zcr = np.mean(lr.feature.zero_crossing_rate(y=augmented).T, axis=0)
#         feature_vector = np.hstack([mfccs, mel_spec, zcr])
#         features.append(feature_vector)
#     return np.array(features)

def preprocess_audio(file_path):
    data, sample_rate = lr.load(file_path, sr=16000)

    augmented_data = augment_audio(data, sample_rate)

    features = []
    for augmented in augmented_data:
        mfccs = np.mean(lr.feature.mfcc(y=augmented, sr=sample_rate).T, axis=0)
        mel_spec = np.mean(lr.feature.melspectrogram(y=augmented, sr=sample_rate).T, axis=0)
        zcr = np.mean(lr.feature.zero_crossing_rate(y=augmented).T, axis=0)
        feature_vector = np.hstack([mfccs, mel_spec, zcr])
        features.append(feature_vector)

    return np.array(features)

def classify_audio(file_path):
    try:
        print(file_path)
        X_new = preprocess_audio(file_path)
        print(X_new.shape)

        X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
        print(X_new.shape)

        predictions = model.predict(X_new)
        print(predictions.shape)

        average_prediction = np.mean(predictions, axis=0)
        print(average_prediction)
        predicted_label = np.argmax(average_prediction)
        print(predicted_label)
        predicted_percentage = average_prediction[predicted_label] * 100
        print(f"Nhãn dự đoán: {CLASS_MAPPING[predicted_label]} ({predicted_label}) với xác suất: {predicted_percentage:.2f}%")
        return {
            "transcript": CLASS_MAPPING[predicted_label],  # Return the class name instead of the number
            "label": int(predicted_label),                # Also include the numerical label
            "confidence": float(predicted_percentage)     # Ensure confidence is returned as float
        }
    except Exception as e:
        return {"error": str(e)}


@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = classify_audio(file_path)
 
        return jsonify(result)


if __name__ == "__main__":
    # Thread(daemon=True, target=MetricsExporter(service_key="simple-flask", push_gateway_url="192.168.0.86:19091",service_port=port).start_collect_and_push_metrics).start()
    port = int(os.environ.get("PORT", 5001))

    # Run the Flask app
    app.run(host='0.0.0.0', port=port)
# if __name__ == '__main__':
#     app.run(debug=True)