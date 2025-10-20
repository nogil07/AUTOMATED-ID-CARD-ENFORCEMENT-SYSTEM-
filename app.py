# app.py
import sys
import os
import threading
import traceback # Import traceback for detailed error logging

# --- Flask and utils ---
from flask import (Flask, request, jsonify, send_file,
                   render_template, current_app)

# --- Local Module Imports ---
# Ensure these files exist in the same directory or are accessible via PYTHONPATH
try:
    from config_loader import load_config
    from model_loader import load_models # Loads YOLO models and InsightFace app
    from database_manager import DatabaseManager # Handles DB, Embeddings, and Email triggering
    from image_processor import process_frame_logic # Performs actual frame analysis
    from utils import decode_image, encode_image # Image encoding/decoding helpers
    from fined_log_manager import FinedLogManager
except ImportError as e:
    print(f"FATAL: Failed to import necessary modules: {e}")
    print("Ensure config_loader.py, model_loader.py, database_manager.py, "
          "image_processor.py, email_notifier.py, fined_log_manager.py, "
          "and utils.py are present.")
    sys.exit(1)

# --- Global Variables ---
# Note: While globals work here, using app context (current_app) or dependency injection
#       can be more robust patterns for larger applications.
CONFIG = None
person_model = None
id_card_model = None
face_app = None # Global for InsightFace FaceAnalysis application
db_manager = None # Manages database, embeddings, and email logic
fined_log_manager = None # Manages the fined event log
models_loaded_ok = False # Flag to track if all models loaded successfully

# --- Flask App Initialization ---
# Looks for templates in a 'templates' subfolder by default
app = Flask(__name__)


# --- Initialization Function ---
def initialize_app():
    """Loads configuration, models, and initializes manager instances."""
    # Use global keyword to modify globals defined outside the function
    global CONFIG, person_model, id_card_model, face_app, db_manager
    global fined_log_manager, models_loaded_ok

    print("\n" + "="*60 + "\n      Starting ID Card Compliance Monitoring System\n" + "="*60 + "\n")

    # 1. Load Configuration
    try:
        CONFIG = load_config()
        if not isinstance(CONFIG, dict):
             raise TypeError("load_config did not return a dictionary.")
        print("[ OK ] Configuration loaded.")
    except Exception as e:
        print(f"[FATAL] Failed to load configuration: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Load Models
    try:
        # Unpack all return values
        person_model, id_card_model, face_app, models_loaded_ok = load_models(CONFIG)
        if not models_loaded_ok:
            print("\n[WARNING] One or more models failed to load. Check model paths and dependencies.")
        else:
            print("[ OK ] Detection and Recognition models loaded.")
    except Exception as e:
        print(f"[FATAL] Error during model loading: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Initialize Database Manager
    try:
        db_manager = DatabaseManager(CONFIG)
        if not db_manager.is_loaded:
             print("[WARNING] Database CSV loading failed or encountered issues.")
        # Check if embeddings loaded if recognition models are ready
        _, known_embeddings = db_manager.get_recognition_data()
        if not known_embeddings and face_app is not None:
            print("[WARNING] No known embeddings loaded. Face recognition will be disabled.")
        print("[ OK ] Database Manager initialized.")
    except Exception as e:
         print(f"[CRITICAL WARNING] Failed to initialize DatabaseManager: {e}. Core functionality may fail.")
         traceback.print_exc()
         db_manager = None # Ensure it's None on failure

    # 4. Initialize Fined Log Manager
    try:
        log_csv_path = CONFIG.get('fined_log_csv', 'fined_log.csv')
        log_manager_instance = FinedLogManager(log_csv_path)
        # Attach to app context (preferred access method in routes)
        app.fined_log_manager = log_manager_instance
        # Assign to global as well (as currently used)
        fined_log_manager = log_manager_instance
        print(f"[ OK ] Fined Log Manager initialized (File: {log_csv_path}).")
    except Exception as e:
        print(f"[CRITICAL WARNING] Failed to initialize FinedLogManager: {e}. Fined event logging disabled.")
        traceback.print_exc()
        app.fined_log_manager = None # Ensure app attribute is None on failure
        fined_log_manager = None # Ensure global is None on failure

    # 5. Print Configuration Summary
    print("\n--- Backend Ready ---")
    print(f"Configured Settings Summary:")
    # Use .get with defaults for safer access
    print(f"  - Camera Preference: Index {CONFIG.get('camera_index', 'N/A')}")
    print(f"  - Person Model:      {CONFIG.get('person_model_path', 'N/A')}")
    print(f"  - ID Card Model:     {CONFIG.get('id_card_model_path', 'N/A')}")
    print(f"  - ArcFace Model:     {CONFIG.get('model_name', 'N/A')} (via InsightFace)")
    print(f"  - Student Database:  {CONFIG.get('csv_file', 'N/A')}")
    print(f"  - Embeddings File:   {CONFIG.get('embeddings_file', 'N/A')}") # Corrected Key
    print(f"  - ArcFace Threshold: {CONFIG.get('similarity_threshold', 'N/A')}")
    print(f"  - Fine Amount:       ${CONFIG.get('fine_amount', 0.0):.2f}")
    email_status = "Enabled" if CONFIG.get('email_enabled', False) else "Disabled"
    sender = CONFIG.get('sender_email', 'N/A')
    print(f"  - Email Notifications: {email_status} (Sender: {sender})")
    img_dir = CONFIG.get('fined_images_dir', 'N/A')
    log_csv = CONFIG.get('fined_log_csv', 'N/A')
    print(f"  - Fined Image Dir:   {img_dir}")
    print(f"  - Fined Log CSV:     {log_csv}")
    print("--------------------------")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    try:
        if CONFIG is None:
             app.logger.error("Global CONFIG is None during index route access!")
             return "Error: Application configuration not loaded correctly.", 500

        # Access camera_index directly from the flat CONFIG dictionary
        camera_index = CONFIG.get('camera_index', 0)
        camera_index_str = str(camera_index)

        # Pass the string value to the template
        return render_template('index.html', preferred_camera_index_str=camera_index_str)

    except Exception as e:
        app.logger.error(f"Error rendering index template: {e}", exc_info=True) # Log with traceback
        # Provide a user-friendly error without exposing details
        return "Error loading page.", 500


@app.route('/process', methods=['POST'])
def process_image_endpoint():
    """Receives image data, processes it, and returns results."""
    # Access log manager via application context
    log_manager = current_app.fined_log_manager
    app.logger.debug(f"Checking log_manager from current_app. Is None: {log_manager is None}") # Use Flask logger

    # Check if essential components are loaded and ready
    # Using globals here as they were set up this way, though current_app is often better
    if not models_loaded_ok or person_model is None or id_card_model is None or face_app is None:
         app.logger.error("Processing attempt failed: Core models/apps not loaded.")
         return jsonify({"error": "Core models/apps not loaded", "processed_image": None, "detections": []}), 503
    if db_manager is None or not db_manager.is_loaded:
        app.logger.error("Processing attempt failed: Database unavailable.")
        return jsonify({"error":"Database unavailable"}), 503
    if log_manager is None:
         app.logger.warning("FinedLogManager not available via current_app. Fines will not be logged to CSV.")
         # Continue processing, but logging won't happen

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            app.logger.warning("'/process' request received without image data.")
            return jsonify({"error": "No image data provided"}), 400

        frame = decode_image(data['image'])
        if frame is None:
            app.logger.error("Failed to decode image data in '/process'.")
            return jsonify({"error": "Failed to decode image data"}), 400

        # --- Call the main processing logic ---
        processed_frame, detected_info = process_frame_logic(
            frame, person_model, id_card_model, face_app, db_manager, log_manager, CONFIG
        )

        # Handle potential errors reported by processing logic
        if processed_frame is None:
             error_msg = "Unknown processing error occurred"
             if detected_info and isinstance(detected_info, list) and len(detected_info) > 0 and 'error' in detected_info[0]:
                 error_msg = detected_info[0]['error']
             app.logger.error(f"Error reported by process_frame_logic: {error_msg}")
             return jsonify({"error": error_msg}), 500

        # Encode the processed frame
        encoded_frame = encode_image(processed_frame)
        if encoded_frame is None:
            app.logger.error("Error encoding processed frame to base64.")
            return jsonify({"error": "Failed to encode processed image"}), 500

        # Return successful results
        return jsonify({
            "processed_image": encoded_frame,
            "detections": detected_info
        })

    except Exception as e:
        # Catch-all for unexpected errors within the endpoint
        app.logger.error(f"Unexpected Error in /process endpoint: {e}", exc_info=True) # Log full traceback
        return jsonify({"error": "An internal server error occurred"}), 500 # Generic error to client


@app.route('/get_totals', methods=['GET'])
def get_totals_endpoint():
    """Returns the current violation count and total fine amount."""
    if db_manager is None:
         app.logger.warning("'/get_totals' called but DatabaseManager is not available.")
         return jsonify({"violations": 0, "fine": 0.0}) # Return default if DB manager failed
    try:
        violations, fine = db_manager.get_totals()
        return jsonify({"violations": violations, "fine": float(fine)})
    except Exception as e:
        app.logger.error(f"Error in /get_totals: {e}", exc_info=True)
        return jsonify({"error": "Failed to calculate totals"}), 500


@app.route('/export_violations', methods=['GET'])
def export_violations_endpoint():
    """Exports the student database as a CSV file."""
    if db_manager is None or not db_manager.is_loaded:
        app.logger.error("'/export_violations' called but DatabaseManager is not available.")
        return "Error: Database is not available for export.", 503
    try:
        csv_buffer, filename, mimetype = db_manager.export_database_csv()
        return send_file(
            csv_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except ValueError as ve:
         app.logger.error(f"Export failed (ValueError): {ve}")
         return str(ve), 503
    except Exception as e:
        app.logger.error(f"Error during CSV export: {e}", exc_info=True)
        return "Error generating export file.", 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the initialization sequence
    initialize_app()

    # Check if initialization was successful enough to run
    if CONFIG is None or person_model is None or id_card_model is None or face_app is None or db_manager is None:
         print("\n[FATAL] Application initialization failed. Cannot start server.")
         sys.exit(1)
    # Log manager is optional, so we don't exit if it failed

    # Start the Flask web server
    print("\n--- Starting Flask Server ---")
    print(f"Access UI via: http://127.0.0.1:5000 (or your server's IP)")
    print("Use CTRL+C in the terminal to stop the server.")
    try:
        # Use threaded=True for background tasks like email, but consider a production server for deployment
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
         print(f"\n--- Flask Server Could Not Start: {e} ---")
         traceback.print_exc()
    finally:
        print("\n--- Server Shutdown ---")