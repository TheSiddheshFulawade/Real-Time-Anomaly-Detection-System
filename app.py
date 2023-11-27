from flask import Flask, request, render_template, redirect, url_for, flash, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
from os import path
from flask_login import LoginManager
from flask_login import login_user, login_required, logout_user, current_user
from flask import Flask
from extensions import db 
from database import User
from database import Prediction
from model import predict_on_video, predict_single_action
from model import predict_and_display_live_video

app = Flask(__name__, static_url_path='/static')

# db = SQLAlchemy()
DB_NAME = "database.db"

app.config['SECRET_KEY'] = 'siddhesh'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
db.init_app(app)

from database import User, Note
    
with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')


SEQUENCE_LENGTH = 20
UPLOAD_FOLDER = r'D:\Anomaly Detection\Untitled Folder\test_videos'
OUTPUT_FOLDER = r'static/output_videos'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.add_url_rule('/output_videos/<filename>', 'uploaded_file', build_only=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    from werkzeug.security import check_password_hash
    if request.method == 'POST':
        uname = request.form.get('uname')
        password = request.form.get('password')
        user = User.query.filter_by(uname=uname).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('index'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Login failed. Please check your username and password.', category='error')

    return render_template('login.html', user=current_user)


@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullName')
        email = request.form.get('email')
        uname = request.form.get('uname')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        phone = request.form.get('phone')
        address = request.form.get('address')

        user = User.query.filter_by(uname=uname).first()
        useremail = User.query.filter_by(email=email).first()
        
        if user:
            flash('Username already exists.', category='error')
        elif useremail:
            flash('Email already exists.', category='error')
        elif len(uname) < 8:
            flash('Your Username should be greater than 8 characters.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(phone) != 10:
            flash('Your Phone No. must be 10 digits.', category='error')
        else:
            new_user = User(fullName=fullname, email=email, uname=uname, password=generate_password_hash(
                password1, method='pbkdf2:sha256', salt_length=8), phone=phone, address=address)
            db.session.add(new_user)
            db.session.commit()
            # Now that the user is successfully created, you can log them in
            login_user(new_user, remember=True)
            flash('Account Created Successfully!', category='success')
            return redirect(url_for('home'))

    return render_template('register.html', user=current_user)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_video_path = os.path.join(app.root_path, app.config['OUTPUT_FOLDER'], file.filename)

        file.save(video_path)

        # Process the uploaded video and generate the output video
        predict_on_video(video_path, output_video_path, SEQUENCE_LENGTH)

        # Get predictions from the processed video
        predictions, confidence = predict_single_action(video_path, SEQUENCE_LENGTH)

        # Create a new Prediction record
        new_prediction = Prediction(
            user_id=current_user.id,
            video_file=file.filename,
            predictions=predictions,
            confidence=confidence  # Add the actual confidence value here
        )

        db.session.add(new_prediction)
        db.session.commit()

        return redirect(url_for('result', video_file=file.filename, predictions=predictions, confidence=confidence))


@app.route('/result/<video_file>')
@login_required
def result(video_file):
    # Assuming you have a user object (current_user) from Flask-Login
    user = current_user
    return render_template('result.html', video_file=video_file, predictions=request.args.get('predictions'), confidence=request.args.get('confidence'), user=user)



@app.route('/live_camera')
@login_required
def live_camera():
    # Call the function to display the live camera feed.
    predict_and_display_live_video(SEQUENCE_LENGTH)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

