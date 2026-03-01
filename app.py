from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from datetime import datetime
import cloudinary
import cloudinary.uploader

# --- App Initialization and Configuration ---
app = Flask(__name__, template_folder="templates", static_folder="static")
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Cloudinary Config ---
app.config['CLOUDINARY_CLOUD_NAME'] = 'daoj7y9ex'
app.config['CLOUDINARY_API_KEY'] = '536481798813966'
app.config['CLOUDINARY_API_SECRET'] = 'kd3_kxALvWElfmR2BuPlJto-Bug'

cloudinary.config(
    cloud_name = app.config['CLOUDINARY_CLOUD_NAME'],
    api_key = app.config['CLOUDINARY_API_KEY'],
    api_secret = app.config['CLOUDINARY_API_SECRET']
)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# --- Database Model Definitions ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    garden_plants = db.relationship('MyGardenPlant', backref='owner', lazy=True)

    def __repr__(self):
        return f"User('{self.username}')"

class MyGardenPlant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant_id = db.Column(db.Integer, nullable=False) 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    nickname = db.Column(db.String(100), nullable=False, default='My Plant')
    date_added = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    growth_entries = db.relationship('GrowthEntry', backref='plant', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"MyGardenPlant('{self.nickname}')"

class GrowthEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    week_number = db.Column(db.Integer, nullable=False)
    photo_url = db.Column(db.String(200), nullable=False) 
    user_notes = db.Column(db.Text, nullable=True)
    date_added = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    my_garden_plant_id = db.Column(db.Integer, db.ForeignKey('my_garden_plant.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Data and Model Loading ---
all_plant_data = {}
try:
    with open(os.path.join(app.static_folder, 'data', 'data.json'), 'r') as f:
        all_plant_data = json.load(f)
    print("Plant data from data.json loaded successfully.")
except Exception as e:
    print(f"Error loading data.json: {e}")

MODEL_PATH = "plant_disease_model.keras"
CLASS_INDICES_PATH = "class_indices.json"
# IMPORTANT: Changed to (256, 256) to fix common model shape mismatch errors.
IMG_SIZE = (256, 256)

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    CLASS_LABELS = {v: k for k, v in class_indices.items()}
    print("Model and class labels loaded successfully.")
except Exception as e:
    print("Error loading model or class labels:", e)
    model = None
    CLASS_LABELS = None

SOLUTIONS = {
    "bell_pepper_bacterial_spot": "Use copper-based fungicide sprays and avoid overhead watering.",
    "bell_pepper_healthy": "No disease detected. Maintain proper watering and fertilization.",
    "chilli_healthy": "Plant is healthy. Ensure balanced fertilizers and pest monitoring.",
    "chilli_leaf_curl": "Control whiteflies/aphids with neem oil or insecticides.",
    "chilli_leaf_spot": "Apply fungicides and remove infected leaves.",
    "chilli_whitefly": "Spray neem oil or imidacloprid for whitefly control.",
    "chilli_yellowish": "Use nitrogen-rich fertilizers and improve soil health.",
    "lemon_anthracnose": "Remove infected leaves and apply copper-based fungicides.",
    "lemon_citrus_canker": "Use resistant varieties, apply copper sprays, and prune affected areas.",
    "lemon_healthy": "No issues found! Keep monitoring your plant regularly.",
    "lemon_sooty_mould": "Wash leaves with mild soap solution and control sap-sucking insects.",
    "tomato_bacterial_spot": "Use disease-free seeds, apply copper sprays, and rotate crops.",
    "tomato_early_blight": "Apply fungicides like chlorothalonil or mancozeb.",
    "tomato_healthy": "Tomato plant is healthy. Continue good practices.",
    "tomato_late_blight": "Spray with metalaxyl fungicide and destroy infected plants.",
    "tomato_leaf_mold": "Improve air circulation, avoid leaf wetness, apply fungicide.",
    "tomato_mosaic_virus": "Remove infected plants, disinfect tools, and control aphids.",
    "tomato_septoria_leaf_spot": "Remove infected leaves, use fungicide sprays.",
    "tomato_spider_mites": "Spray with neem oil or miticide to control mites.",
    "tomato_target_spot": "Remove affected leaves, use fungicide treatment.",
    "tomato_yellow_leaf_curl_virus": "Control whiteflies and use resistant varieties."
}

# --- Route Definitions ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/my-garden")
@login_required
def my_garden():
    user_garden_plants = MyGardenPlant.query.filter_by(owner=current_user).order_by(MyGardenPlant.date_added.desc()).all()
    
    # Safely convert keys to strings to ensure matching works perfectly
    all_plants_dict = {str(plant['id']): plant for plant in all_plant_data.get('floral', []) + all_plant_data.get('edible', [])}
    
    garden_data = []
    for plant in user_garden_plants:
        static_data = all_plants_dict.get(str(plant.plant_id))
        if static_data:
            garden_data.append({
                'my_plant_id': plant.id, 
                'nickname': plant.nickname,
                'date_added': plant.date_added.strftime('%B %d, %Y'),
                'static_name': static_data['name'],
                'static_image': static_data['image'],
                'static_id': plant.plant_id
            })
            
    return render_template("my_garden.html", my_plants=garden_data)

@app.route("/add-to-garden", methods=['POST'])
@login_required
def add_to_garden():
    data = request.get_json()
    raw_plant_id = data.get('plant_id')

    if not raw_plant_id:
        return jsonify({'status': 'error', 'message': 'Plant ID missing.'}), 400

    # Robust handling of ID types
    try:
        plant_id = int(raw_plant_id)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid Plant ID format.'}), 400
    
    default_name = "Plant"
    
    # Safely find the plant name matching the ID
    for cat in all_plant_data:
        for p in all_plant_data[cat]:
            if str(p['id']) == str(plant_id):
                default_name = p['name']
                break
    
    nickname = data.get('nickname', f"My {default_name}")

    new_item = MyGardenPlant(plant_id=plant_id, owner=current_user, nickname=nickname)
    db.session.add(new_item)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': f'{default_name} added to your garden!'})

@app.route("/remove-from-garden", methods=['POST'])
@login_required
def remove_from_garden():
    data = request.get_json()
    my_plant_id = data.get('my_plant_id') 
    
    if not my_plant_id:
        return jsonify({'status': 'error', 'message': 'Plant ID missing.'}), 400
        
    item_to_remove = MyGardenPlant.query.get(my_plant_id)
    
    if item_to_remove and item_to_remove.user_id == current_user.id:
        db.session.delete(item_to_remove)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Plant removed from your garden.'})
        
    return jsonify({'status': 'error', 'message': 'Plant not found or permission denied.'}), 404

@app.route("/get-user-garden-ids")
@login_required
def get_user_garden_ids():
    garden_items = MyGardenPlant.query.filter_by(owner=current_user).all()
    plant_ids = list(set([item.plant_id for item in garden_items]))
    return jsonify(plant_ids)

@app.route("/journal/<int:my_plant_id>")
@login_required
def plant_journal(my_plant_id):
    my_plant = MyGardenPlant.query.get_or_404(my_plant_id)
    
    if my_plant.user_id != current_user.id:
        flash('You do not have permission to view this journal.', 'danger')
        return redirect(url_for('my_garden'))
    
    plant_info = None
    for cat in all_plant_data:
        for p in all_plant_data[cat]:
            if str(p['id']) == str(my_plant.plant_id):
                plant_info = p
                break
                
    entries = GrowthEntry.query.filter_by(my_garden_plant_id=my_plant_id).order_by(GrowthEntry.week_number.asc()).all()
    
    return render_template("plant_journal.html", my_plant=my_plant, plant_info=plant_info, entries=entries)

@app.route("/add-journal-entry", methods=['POST'])
@login_required
def add_journal_entry():
    my_plant_id = request.form.get('my_plant_id')
    week_number = request.form.get('week_number')
    notes = request.form.get('notes')
    photo = request.files.get('photo')

    my_plant = MyGardenPlant.query.get_or_404(my_plant_id)
    if my_plant.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Permission denied.'}), 403

    if not photo:
        return jsonify({'status': 'error', 'message': 'Photo is required.'}), 400

    try:
        upload_result = cloudinary.uploader.upload(photo)
        photo_url = upload_result.get('secure_url')

        new_entry = GrowthEntry(
            week_number=int(week_number),
            photo_url=photo_url,
            user_notes=notes,
            my_garden_plant_id=my_plant.id
        )
        db.session.add(new_entry)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Journal entry added!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/delete-journal-entry/<int:entry_id>", methods=['DELETE'])
@login_required
def delete_journal_entry(entry_id):
    entry = GrowthEntry.query.get_or_404(entry_id)
    
    if entry.plant.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Permission denied.'}), 403
        
    try:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Entry deleted.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or CLASS_LABELS is None:
        return jsonify({"error": "Model or class labels not loaded."}), 500
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)
        class_idx = int(np.argmax(preds[0]))
        predicted_class_name = CLASS_LABELS.get(class_idx, "Unknown")
        solution = SOLUTIONS.get(predicted_class_name, "No solution available.")
        return jsonify({"prediction": predicted_class_name, "solution": solution})
    except Exception as e:
        print("Prediction error:", str(e)) # Added this so it shows in your VS Code terminal!
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)