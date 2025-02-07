from flask import Flask, flash, request, redirect, url_for, render_template
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import joblib
import imutils
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load Models
covid_model = load_model('models/covid.h5')
breastcancer_model = joblib.load('models/breast_cancer_model.pkl')
alzheimer_model = load_model('models/alzheimer_model.h5')
pneumonia_model = load_model('models/pneumonia_model.h5')

# Configure Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret_key"

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

############################################# BRAIN TUMOR FUNCTIONS ################################################

def preprocess_images(image_set, img_size):
    """Resize and apply VGG-16 preprocessing to the images."""
    processed_images = []
    for img in image_set:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        processed_images.append(preprocess_input(img))
    return np.array(processed_images)

def crop_images(image_set, add_pixels=0):
    """Crop the images based on contours to focus on the region of interest."""
    cropped_images = []
    for img in image_set:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding and morphological operations
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours and crop the image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get extreme points to define cropping rectangle
        ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        ext_bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        # Crop the image
        cropped_image = img[ext_top[1]-add_pixels:ext_bottom[1]+add_pixels,
                             ext_left[0]-add_pixels:ext_right[0]+add_pixels].copy()
        cropped_images.append(cropped_image)

    return np.array(cropped_images)

########################### Routing Functions ########################################

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('homepage.html')

@app.route('/covid.html')
def covid():
    """Render the COVID-19 page."""
    return render_template('covid.html')

@app.route('/services.html')
def services():
    """Render the services page."""
    return render_template('services.html')

@app.route('/breastcancer.html')
def breast_cancer():
    """Render the breast cancer page."""
    return render_template('breastcancer.html')

@app.route('/contact.html')
def contact():
    """Render the contact page."""
    return render_template('contact.html')

@app.route('/alzheimer.html')
def alzheimer():
    """Render the Alzheimer page."""
    return render_template('alzheimer.html')

@app.route('/pneumonia.html')
def pneumonia():
    """Render the pneumonia page."""
    return render_template('pneumonia.html')

@app.route('/about.html')
def about():
    """Render the about page."""
    return render_template('about.html')

########################### Result Functions ########################################

@app.route('/resultc', methods=['POST'])
def result_covid():
    """Process COVID-19 test results."""
    if request.method == 'POST':
        user_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'email': request.form['email'],
            'phone': request.form['phone'],
            'gender': request.form['gender'],
            'age': request.form['age']
        }
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.resize(img, (224, 224)) / 255.0
            img = img.reshape(1, 224, 224, 3)
            prediction = covid_model.predict(img)
            result = 1 if prediction >= 0.5 else 0
            return render_template('resultc.html', filename=filename, **user_data, r=result)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/resultbc', methods=['POST'])
def result_breast_cancer():
    """Process breast cancer test results."""
    if request.method == 'POST':
        user_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'age': request.form['age'],
            'gender': request.form['gender']
        }
        features = [
            float(request.form['concave_points_mean']),
            float(request.form['area_mean']),
            float(request.form['radius_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['concavity_mean'])
        ]
        prediction = breastcancer_model.predict(np.array(features).reshape(1, -1))
        return render_template('resultbc.html', **user_data, r=prediction[0])

@app.route('/resulta', methods=['POST'])
def result_alzheimer():
    """Process Alzheimer test results."""
    if request.method == 'POST':
        user_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'age': request.form['age'],
            'gender': request.form['gender']
        }
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.resize(img, (176, 176)) / 255.0
            img = img.reshape(1, 176, 176, 3)
            prediction = alzheimer_model.predict(img)
            predicted_class = prediction[0].argmax()
            return render_template('resulta.html', filename=filename, **user_data, r=predicted_class)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')

@app.route('/resultp', methods=['POST'])
def result_pneumonia():
    """Process pneumonia test results."""
    if request.method == 'POST':
        user_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'age': request.form['age'],
            'gender': request.form['gender']
        }
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.resize(img, (150, 150)) / 255.0
            img = img.reshape(1, 150, 150, 3)
            prediction = pneumonia_model.predict(img)
            result = 1 if prediction >= 0.5 else 0
            return render_template('resultp.html', filename=filename, **user_data, r=result)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """Add headers to control caching and compatibility."""
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(debug=True)
