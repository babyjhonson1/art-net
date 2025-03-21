from flask import Flask, request, send_file, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
from unet_res import *
import torch

app = Flask(__name__)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('Unet_res1.pth', map_location=device))
model.eval()
model = model.to(device)

def predict_matrix(matrix):
    input_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_matrix = output_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
    return output_matrix

# Функция для создания визуализации
def create_visualization(matrix_before, matrix_after):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(matrix_before, cmap='gray', vmin=np.min(matrix_before), vmax=np.max(matrix_before))
    ax1.set_title("Artifacts")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(matrix_after, cmap='gray', vmin=np.min(matrix_after), vmax=np.max(matrix_after))
    ax2.set_title("Output")
    plt.colorbar(im2, ax=ax2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        matrix = pd.read_csv(file, header=None).to_numpy()
        
        if matrix.shape[0] != 512 or matrix.shape[1] != 512:
            return jsonify({"error": "H and W should be 512"}), 400

        result_matrix = predict_matrix(matrix)

        result_csv = pd.DataFrame(result_matrix).to_csv(index=False, header=False)
        result_csv_buf = io.BytesIO(result_csv.encode('utf-8'))

        image_buf = create_visualization(matrix, result_matrix)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("result.csv", result_csv_buf.getvalue())
            zf.writestr("visualization.png", image_buf.getvalue())
        zip_buf.seek(0)

        return send_file(
            zip_buf,
            mimetype='application/zip',
            as_attachment=True,
            download_name='result.zip'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)