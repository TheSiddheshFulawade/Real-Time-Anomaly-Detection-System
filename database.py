from extensions import db 
from flask_login import UserMixin
from sqlalchemy.sql import func

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    fullName = db.Column(db.String(150)) 
    email = db.Column(db.String(150), unique=True)
    uname = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    phone = db.Column(db.String(150))
    address = db.Column(db.String(150))
    predictions = db.relationship('Prediction', backref='user')


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_file = db.Column(db.String(255))
    predictions = db.Column(db.String(255))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=func.now())

    # Change the backref name to 'user_predictions'
    user_predictions = db.relationship('User', back_populates='predictions')



