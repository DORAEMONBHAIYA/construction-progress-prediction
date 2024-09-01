from flask import Flask, request, render_template
from inference import predict_progress
from utils import process_image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        floors = int(request.form['floors'])
        laborers = int(request.form['laborers'])
        image = request.files['image']
        image_path = f"data/raw_images/{image.filename}"
        image.save(image_path)
        
        # Process image and predict progress
        image_features = process_image(image_path)
        progress, time_remaining = predict_progress(floors, laborers, image_features)
        
        return render_template('index.html', progress=progress, time_remaining=time_remaining)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
