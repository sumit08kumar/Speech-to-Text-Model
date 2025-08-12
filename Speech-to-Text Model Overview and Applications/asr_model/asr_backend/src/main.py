import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.asr import asr_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

# Blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(asr_bp, url_prefix='/api/asr')

# Ensure database directory exists BEFORE create_all
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(db_dir, 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if not static_folder_path or not os.path.isdir(static_folder_path):
        return "Static folder not configured", 404

    if path:
        req_path = os.path.join(static_folder_path, path)
        if os.path.exists(req_path):
            return send_from_directory(static_folder_path, path)

    index_path = os.path.join(static_folder_path, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_folder_path, 'index.html')

    return "index.html not found", 404

if __name__ == '__main__':
    # Match your curl (port 5001)
    app.run(host='0.0.0.0', port=5001, debug=True)

