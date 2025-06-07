import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
import pickle
import os
import calendar
from PIL import Image
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import calendar
import sqlite3
import json
import io

# Initialize session state
if 'workers_db' not in st.session_state:
    st.session_state.workers_db = {}
if 'attendance_records' not in st.session_state:
    st.session_state.attendance_records = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_recognition' not in st.session_state:
    st.session_state.last_recognition = {}
if 'recognition_cooldown' not in st.session_state:
    st.session_state.recognition_cooldown = 10  # seconds
if 'video_frame' not in st.session_state:
    st.session_state.video_frame = None
if 'recognition_status' not in st.session_state:
    st.session_state.recognition_status = ""

# File path for database persistence
DATABASE_FILE = "worker_tracking.db"

def init_database():
    """Initialize the SQLite database with necessary tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create workers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        position TEXT,
        shift_type TEXT,
        expected_start_time TEXT,
        face_features BLOB,
        photo BLOB
    )
    ''')
    
    # Create attendance records table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        worker_id TEXT,
        name TEXT,
        department TEXT,
        position TEXT,
        shift_type TEXT,
        expected_time TEXT,
        date TEXT,
        clock_in_time TEXT,
        status TEXT,
        lateness_minutes INTEGER,
        timestamp TEXT,
        FOREIGN KEY (worker_id) REFERENCES workers (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def remove_old_files():
    """Remove old CSV and pickle files if they exist"""
    old_files = ["workers_database.pkl", "attendance_records.csv"]
    for file in old_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                st.sidebar.success(f"Removed old file: {file}")
            except Exception as e:
                st.sidebar.warning(f"Could not remove {file}: {e}")

def load_data():
    """Load workers database and attendance records from SQLite database"""
    # Initialize database if it doesn't exist
    init_database()
    
    # Clear current session state
    st.session_state.workers_db = {}
    st.session_state.attendance_records = []
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Load workers
        cursor.execute("SELECT * FROM workers")
        workers_rows = cursor.fetchall()
        
        for row in workers_rows:
            worker_id = row[0]
            worker_data = {
                'id': row[0],
                'name': row[1],
                'department': row[2],
                'position': row[3],
                'shift_type': row[4],
                'expected_start_time': row[5]
            }
            
            # Deserialize face features and photo if they exist
            if row[6]:  # face_features
                worker_data['face_features'] = pickle.loads(row[6])
            
            if row[7]:  # photo
                worker_data['photo'] = pickle.loads(row[7])
                
            st.session_state.workers_db[worker_id] = worker_data
        
        # Load attendance records
        cursor.execute("SELECT * FROM attendance")
        attendance_rows = cursor.fetchall()
        
        for row in attendance_rows:
            attendance_record = {
                'worker_id': row[1],
                'name': row[2],
                'department': row[3],
                'position': row[4],
                'shift_type': row[5],
                'expected_time': row[6],
                'date': row[7],
                'clock_in_time': row[8],
                'status': row[9],
                'lateness_minutes': row[10],
                'timestamp': row[11]
            }
            st.session_state.attendance_records.append(attendance_record)
        
        conn.close()
            
    except Exception as e:
        st.error(f"Error loading data from database: {e}")

def save_data():
    """Save workers database and attendance records to SQLite database"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Clear existing data (optional, could also just update/insert)
        cursor.execute("DELETE FROM workers")
        
        # Save workers
        for worker_id, worker_data in st.session_state.workers_db.items():
            # Serialize complex objects
            face_features_blob = None
            if 'face_features' in worker_data:
                face_features_blob = pickle.dumps(worker_data['face_features'])
            
            photo_blob = None
            if 'photo' in worker_data:
                photo_blob = pickle.dumps(worker_data['photo'])
            
            cursor.execute('''
            INSERT INTO workers (id, name, department, position, shift_type, expected_start_time, face_features, photo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                worker_data['id'],
                worker_data['name'],
                worker_data.get('department', ''),
                worker_data.get('position', ''),
                worker_data.get('shift_type', ''),
                worker_data.get('expected_start_time', ''),
                face_features_blob,
                photo_blob
            ))
        
        # Save attendance records (first clear, then add all)
        cursor.execute("DELETE FROM attendance")
        
        for record in st.session_state.attendance_records:
            cursor.execute('''
            INSERT INTO attendance (worker_id, name, department, position, shift_type, expected_time, date, 
                                    clock_in_time, status, lateness_minutes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['worker_id'],
                record['name'],
                record.get('department', ''),
                record.get('position', ''),
                record.get('shift_type', ''),
                record.get('expected_time', ''),
                record['date'],
                record['clock_in_time'],
                record['status'],
                record['lateness_minutes'],
                record['timestamp']
            ))
        
        conn.commit()
        conn.close()
            
    except Exception as e:
        st.error(f"Error saving data to database: {e}")

def extract_face_features(image):
    """Extract face features using OpenCV"""
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Load OpenCV's pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            return None
        
        # Get the largest face (assuming it's the main subject)
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size for comparison
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Calculate histogram as a simple feature
        hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Normalize histogram
        hist = hist / (hist.sum() + 1e-10)
        
        return {
            'histogram': hist,
            'face_region': face_roi,
            'bbox': (x, y, w, h),
            'faces': faces  # Return all detected faces for drawing
        }
        
    except Exception as e:
        st.error(f"Error extracting face features: {e}")
        return None

def compare_faces(features1, features2, threshold=0.7):
    """Compare two face feature sets"""
    try:
        if features1 is None or features2 is None:
            return False
        
        # Compare histograms using cosine similarity
        hist1 = features1['histogram'].reshape(1, -1)
        hist2 = features2['histogram'].reshape(1, -1)
        
        similarity = cosine_similarity(hist1, hist2)[0][0]
        
        return similarity > threshold
    except Exception as e:
        st.error(f"Error comparing faces: {e}")
        return False

def recognize_worker(image):
    """Recognize worker from image"""
    if not st.session_state.workers_db:
        return None, None
    
    # Get face features for the input image
    input_features = extract_face_features(image)
    if input_features is None:
        return None, None
    
    # Compare with stored features
    best_match = None
    best_similarity = 0
    
    for worker_id, worker_data in st.session_state.workers_db.items():
        stored_features = worker_data['face_features']
        
        # Compare faces
        if compare_faces(input_features, stored_features):
            # Calculate similarity score for ranking
            hist1 = input_features['histogram'].reshape(1, -1)
            hist2 = stored_features['histogram'].reshape(1, -1)
            similarity = cosine_similarity(hist1, hist2)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = worker_data
    
    return best_match, input_features

def can_clock_in(worker_id):
    """Check if worker can clock in (cooldown period)"""
    current_time = time.time()
    if worker_id in st.session_state.last_recognition:
        time_diff = current_time - st.session_state.last_recognition[worker_id]
        return time_diff > st.session_state.recognition_cooldown
    return True

def process_clock_in(worker_data):
    """Process clock-in for a worker"""
    if worker_data and can_clock_in(worker_data['id']):
        # Check if already clocked in today
        today = date.today().strftime("%Y-%m-%d")
        already_clocked = any(
            record['worker_id'] == worker_data['id'] and 
            record['date'] == today 
            for record in st.session_state.attendance_records
        )
        
        if already_clocked:
            st.session_state.recognition_status = f"⚠️ {worker_data['name']} already clocked in today!"
            return False
        else:
            # Record attendance
            clock_in_time = datetime.now().strftime("%H:%M:%S")
            expected_time = worker_data.get('expected_start_time', '08:00:00')
            
            # Determine if the worker is on time or late
            clock_in_dt = datetime.strptime(clock_in_time, "%H:%M:%S")
            expected_dt = datetime.strptime(expected_time, "%H:%M:%S")
            
            # Calculate lateness in minutes
            lateness_minutes = 0
            status = "On Time"
            
            if clock_in_dt > expected_dt:
                lateness_minutes = (clock_in_dt - expected_dt).seconds // 60
                status = "Late"
            
            attendance_record = {
                'worker_id': worker_data['id'],
                'name': worker_data['name'],
                'department': worker_data['department'],
                'position': worker_data['position'],
                'shift_type': worker_data.get('shift_type', 'Not specified'),
                'expected_time': expected_time,
                'date': today,
                'clock_in_time': clock_in_time,
                'status': status,
                'lateness_minutes': lateness_minutes,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.attendance_records.append(attendance_record)
            st.session_state.last_recognition[worker_data['id']] = time.time()
            save_data()
            
            status_emoji = "✅" if status == "On Time" else "⚠️"
            status_text = f"{status_emoji} {worker_data['name']} clocked in at {clock_in_time}"
            if status == "Late":
                status_text += f" ({lateness_minutes} minutes late)"
            
            st.session_state.recognition_status = status_text
            return True
    return False

def draw_face_boxes(image, faces, recognized_worker=None):
    """Draw face detection boxes on image"""
    if faces is not None and len(faces) > 0:
        for (x, y, w, h) in faces:
            if recognized_worker:
                # Green box for recognized worker
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, recognized_worker['name'], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Red box for unrecognized face
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(image, "Unknown", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.running = False
        
    def start_camera(self, camera_index=0):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                # Try fallback to other camera indices if the first one fails
                for idx in range(1, 3):  # Try camera indices 1 and 2
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        break
                
                if not self.cap.isOpened():
                    return False
                    
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            return True
        except Exception as e:
            st.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        """Get a frame from the camera"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        return None

def realtime_clock_in_page():
    """Real-time clock in page"""
    st.header("🎥 Real-time Worker Recognition")
    st.write("Stand in front of the camera for automatic recognition and clock-in.")
    
    # Initialize video processor
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1, 
                                       help="Try different indices if camera doesn't start (0-5)")
        if st.button("📹 Start Camera", key="start_camera"):
            if st.session_state.video_processor.start_camera(camera_index):
                st.session_state.camera_active = True
                st.success("Camera started!")
            else:
                st.error("Failed to start camera")
    
    with col2:
        if st.button("⏹️ Stop Camera", key="stop_camera"):
            st.session_state.video_processor.stop_camera()
            st.session_state.camera_active = False
            st.session_state.video_frame = None
            st.info("Camera stopped")
    
    with col3:
        cooldown = st.number_input("Recognition Cooldown (seconds)", 
                                 min_value=5, max_value=60, 
                                 value=st.session_state.recognition_cooldown,
                                 help="Minimum time between recognitions for the same worker")
        st.session_state.recognition_cooldown = cooldown
    
    # Display status
    if st.session_state.recognition_status:
        if "✅" in st.session_state.recognition_status:
            st.success(st.session_state.recognition_status)
        elif "⚠️" in st.session_state.recognition_status:
            st.warning(st.session_state.recognition_status)
        else:
            st.info(st.session_state.recognition_status)
    
    # Video display and processing
    if st.session_state.camera_active:
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Real-time processing loop
        while st.session_state.camera_active:
            frame = st.session_state.video_processor.get_frame()
            
            if frame is not None:
                # Process frame for recognition
                worker_data, face_features = recognize_worker(frame)
                
                # Draw face boxes
                if face_features and 'faces' in face_features:
                    frame_with_boxes = draw_face_boxes(frame.copy(), face_features['faces'], worker_data)
                else:
                    frame_with_boxes = frame
                
                # Display frame
                video_placeholder.image(frame_with_boxes, channels="RGB", use_container_width=True)
                
                # Process recognition
                if worker_data:
                    with info_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**👤 Recognized Worker:**")
                            st.write(f"**Name:** {worker_data['name']}")
                            st.write(f"**ID:** {worker_data['id']}")
                            st.write(f"**Department:** {worker_data['department']}")
                        
                        with col2:
                            if 'photo' in worker_data:
                                st.image(worker_data['photo'], caption="Registered Photo", width=150)
                    
                    # Auto clock-in
                    if process_clock_in(worker_data):
                        # Show success animation
                        st.balloons()
                        time.sleep(2)  # Brief pause after successful clock-in
                else:
                    info_placeholder.info("👀 Looking for registered workers...")
                
            else:
                st.error("Failed to get camera frame")
                break
            
            # Small delay to prevent excessive processing
            time.sleep(0.1)
    
    else:
        st.info("Click 'Start Camera' to begin real-time recognition")
        
        # Show sample of what will be detected
        if st.session_state.workers_db:
            st.subheader("📋 Registered Workers")
            cols = st.columns(min(3, len(st.session_state.workers_db)))
            for idx, (worker_id, worker_data) in enumerate(st.session_state.workers_db.items()):
                with cols[idx % 3]:
                    st.write(f"**{worker_data['name']}**")
                    st.write(f"ID: {worker_data['id']}")
                    if 'photo' in worker_data:
                        st.image(worker_data['photo'], width=100)

def add_worker_page():
    """Page for adding new workers"""
    st.header("➕ Add New Worker")
    
    # Initialize video processor for camera capture if needed
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    
    # Initialize worker photo variables
    if 'worker_captured_photo' not in st.session_state:
        st.session_state.worker_captured_photo = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        worker_name = st.text_input("Worker Name")
        worker_id = st.text_input("Worker ID")
        department = st.text_input("Department")
        position = st.text_input("Position")
        shift_type = st.selectbox(
            "Shift Type", 
            options=["Day Shift (8:00 AM)", "Night Shift (6:00 PM)"],
            help="Day shift workers start at 8:00 AM, Night shift workers start at 6:00 PM"
        )
    
    with col2:
        # Option 1: Upload photo
        st.subheader("Option 1: Upload Photo")
        uploaded_file = st.file_uploader("Upload Worker Photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Photo", use_container_width=True)
        
        # Option 2: Take photo with camera
        st.subheader("Option 2: Take Photo")
        camera_tab1, camera_tab2 = st.tabs(["Streamlit Camera", "Live Camera"])
        
        with camera_tab1:
            camera_photo = st.camera_input("Take a photo")
        
        with camera_tab2:
            # Live camera controls
            camera_col1, camera_col2 = st.columns(2)
            
            with camera_col1:
                camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1, 
                                              help="Try different indices if camera doesn't start", key="add_worker_camera_index")
                
                if st.button("📹 Start Camera", key="add_worker_start_camera"):
                    if st.session_state.video_processor.start_camera(camera_index):
                        st.session_state.camera_active = True
                        st.success("Camera started!")
                    else:
                        st.error("Failed to start camera")
            
            with camera_col2:
                if st.button("⏹️ Stop Camera", key="add_worker_stop_camera"):
                    st.session_state.video_processor.stop_camera()
                    st.session_state.camera_active = False
                    st.session_state.worker_captured_photo = None
                    st.info("Camera stopped")
                
                if st.button("📸 Capture Photo", key="add_worker_capture_photo", disabled=not st.session_state.camera_active):
                    if st.session_state.camera_active:
                        frame = st.session_state.video_processor.get_frame()
                        if frame is not None:
                            st.session_state.worker_captured_photo = frame
                            st.success("Photo captured!")
                        else:
                            st.error("Failed to capture photo")
            
            # Display live camera or captured photo
            if st.session_state.camera_active and not st.session_state.worker_captured_photo is not None:
                # Show live camera feed
                camera_placeholder = st.empty()
                frame = st.session_state.video_processor.get_frame()
                if frame is not None:
                    camera_placeholder.image(frame, channels="RGB", use_container_width=True, caption="Live Camera")
            
            # Display captured photo if available
            if st.session_state.worker_captured_photo is not None:
                st.image(st.session_state.worker_captured_photo, channels="RGB", use_container_width=True, caption="Captured Photo")
        
        if camera_photo is not None:
            camera_image = Image.open(camera_photo)
            st.image(camera_image, caption="Camera Photo", use_container_width=True)
    
    if st.button("Add Worker"):
        # Determine which image to use
        selected_image = None
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
        elif camera_photo is not None:
            selected_image = Image.open(camera_photo)
        elif st.session_state.get('worker_captured_photo') is not None:
            # Convert numpy array to PIL Image
            selected_image = Image.fromarray(st.session_state.worker_captured_photo)
        
        if worker_name and worker_id and selected_image is not None:
            if worker_id in st.session_state.workers_db:
                st.error("Worker ID already exists!")
            else:
                # Extract face features
                face_features = extract_face_features(selected_image)
                
                if face_features is not None:
                    # Store worker data
                    st.session_state.workers_db[worker_id] = {
                        'name': worker_name,
                        'id': worker_id,
                        'department': department,
                        'position': position,
                        'shift_type': shift_type,
                        'expected_start_time': '08:00:00' if 'Day Shift' in shift_type else '18:00:00',
                        'face_features': face_features,
                        'photo': np.array(selected_image)
                    }
                    
                    save_data()
                    st.success(f"Worker {worker_name} added successfully!")
                    st.balloons()
                else:
                    st.error("No face detected in the image. Please upload a clear photo with a visible face.")
        else:
            st.error("Please fill in all required fields and provide a photo (upload or camera).")

def attendance_records_page():
    """Page for viewing attendance records"""
    st.header("📊 Attendance Records")
    
    if st.session_state.attendance_records:
        df = pd.DataFrame(st.session_state.attendance_records)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_date = st.date_input("Filter by Date", value=date.today())
        with col2:
            departments = ["All"] + list(set([record['department'] for record in st.session_state.attendance_records if record['department']]))
            selected_dept = st.selectbox("Filter by Department", departments)
        with col3:
            if st.button("Clear Filters"):
                st.rerun()
        
        # Apply filters
        filtered_df = df.copy()
        if selected_date:
            filtered_df = filtered_df[filtered_df['date'] == selected_date.strftime("%Y-%m-%d")]
        if selected_dept != "All":
            filtered_df = filtered_df[filtered_df['department'] == selected_dept]
        
        # Display records
        if not filtered_df.empty:
            st.dataframe(filtered_df[['name', 'worker_id', 'department', 'position', 'shift_type', 'date', 'clock_in_time', 'status', 'lateness_minutes']], use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"attendance_records_{selected_date}.csv",
                mime="text/csv"
            )
        else:
            st.info("No records found for the selected filters.")
    else:
        st.info("No attendance records available.")

def workers_management_page():
    """Page for managing workers"""
    st.header("👥 Workers Management")
    
    if st.session_state.workers_db:
        st.subheader("Registered Workers")
        
        # Search functionality
        search_term = st.text_input("🔍 Search workers by name or ID")
        
        filtered_workers = st.session_state.workers_db
        if search_term:
            filtered_workers = {
                k: v for k, v in st.session_state.workers_db.items()
                if search_term.lower() in v['name'].lower() or search_term.lower() in v['id'].lower()
            }
        
        for worker_id, worker_data in filtered_workers.items():
            with st.expander(f"{worker_data['name']} ({worker_id})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {worker_data['name']}")
                    st.write(f"**ID:** {worker_data['id']}")
                    st.write(f"**Department:** {worker_data['department']}")
                    st.write(f"**Position:** {worker_data['position']}")
                    st.write(f"**Shift Type:** {worker_data.get('shift_type', 'Not specified')}")
                    st.write(f"**Expected Start Time:** {worker_data.get('expected_start_time', 'Not specified')}")
                    
                    if st.button(f"🗑️ Delete {worker_data['name']}", key=f"delete_{worker_id}"):
                        del st.session_state.workers_db[worker_id]
                        save_data()
                        st.success(f"Worker {worker_data['name']} deleted!")
                        st.rerun()
                
                with col2:
                    if 'photo' in worker_data:
                        st.image(worker_data['photo'], caption="Worker Photo", width=200)
        
        if not filtered_workers:
            st.info("No workers found matching the search criteria.")
    else:
        st.info("No workers registered yet.")

def attendance_statistics_page():
    """Page for viewing attendance statistics"""
    st.header("📈 Attendance Statistics")
    
    if not st.session_state.attendance_records:
        st.info("No attendance records available for statistics.")
        return
    
    # Create DataFrame from attendance records
    df = pd.DataFrame(st.session_state.attendance_records)
    
    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Add week and month columns
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.year
    
    # Time period selection
    time_period = st.radio(
        "Select Time Period",
        ["Weekly", "Monthly"],
        horizontal=True
    )
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if time_period == "Weekly":
            # Get all available weeks
            weeks = sorted(df['week'].unique())
            selected_week = st.selectbox("Select Week", weeks, index=len(weeks)-1)
            filtered_df = df[df['week'] == selected_week]
        else:  # Monthly
            # Get all available months
            months = sorted([(y, m) for y, m in zip(df['year'], df['month'])], reverse=True)
            month_options = [f"{calendar.month_name[m]} {y}" for y, m in months]
            selected_month_idx = st.selectbox("Select Month", range(len(month_options)), format_func=lambda x: month_options[x])
            selected_year, selected_month = months[selected_month_idx]
            filtered_df = df[(df['month'] == selected_month) & (df['year'] == selected_year)]
    
    with col2:
        departments = ["All"] + sorted(df['department'].unique().tolist())
        selected_dept = st.selectbox("Department", departments)
        
        if selected_dept != "All":
            filtered_df = filtered_df[filtered_df['department'] == selected_dept]
    
    with col3:
        workers = ["All Workers"] + sorted(df['name'].unique().tolist())
        selected_worker = st.selectbox("Worker", workers)
        
        if selected_worker != "All Workers":
            filtered_df = filtered_df[filtered_df['name'] == selected_worker]
    
    if filtered_df.empty:
        st.info(f"No records found for the selected {time_period.lower()} period.")
        return
    
    # Display basic statistics
    st.subheader("Summary Statistics")
    
    total_records = len(filtered_df)
    on_time_count = len(filtered_df[filtered_df['status'] == 'On Time'])
    late_count = len(filtered_df[filtered_df['status'] == 'Late'])
    
    on_time_percentage = (on_time_count / total_records) * 100 if total_records > 0 else 0
    late_percentage = (late_count / total_records) * 100 if total_records > 0 else 0
    
    # Create metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Attendance", total_records)
    col2.metric("On Time", f"{on_time_count} ({on_time_percentage:.1f}%)")
    col3.metric("Late", f"{late_count} ({late_percentage:.1f}%)")
    
    # Average lateness
    if 'lateness_minutes' in filtered_df.columns and late_count > 0:
        avg_lateness = filtered_df[filtered_df['status'] == 'Late']['lateness_minutes'].mean()
        st.metric("Average Lateness (minutes)", f"{avg_lateness:.1f}")
    
    # Single worker detailed statistics
    if selected_worker != "All Workers":
        st.markdown("---")
        st.subheader(f"📊 Detailed Statistics for {selected_worker}")
        
        # Get all records for this worker (not just the filtered period)
        worker_all_records = df[df['name'] == selected_worker]
        
        # Get worker information
        worker_info = None
        if st.session_state.workers_db:
            # Find the worker in the database
            for worker_id, worker_data in st.session_state.workers_db.items():
                if worker_data['name'] == selected_worker:
                    worker_info = worker_data
                    break
        
        # Show worker details if available
        if worker_info:
            info_cols = st.columns(4)
            info_cols[0].info(f"**Department:** {worker_info.get('department', 'Not specified')}")
            info_cols[1].info(f"**Position:** {worker_info.get('position', 'Not specified')}")
            info_cols[2].info(f"**Shift Type:** {worker_info.get('shift_type', 'Not specified')}")
            info_cols[3].info(f"**Expected Start:** {worker_info.get('expected_start_time', 'Not specified')}")
            
            # Display worker photo if available
            if 'photo' in worker_info and worker_info['photo'] is not None:
                with st.expander("Show Worker Photo"):
                    st.image(worker_info['photo'], width=200, caption=f"{selected_worker}'s Photo")
        
        # View selector for the individual worker stats
        worker_view = st.radio(
            "Select View",
            ["Summary", "Attendance Calendar", "Trends & Patterns", "Monthly Analysis"],
            horizontal=True
        )
        
        # Calculate totals (needed for all views)
        total_days = len(worker_all_records)
        on_time_days = len(worker_all_records[worker_all_records['status'] == 'On Time'])
        late_days = len(worker_all_records[worker_all_records['status'] == 'Late'])
        
        if worker_view == "Summary":
            # Create summary metrics
            st.subheader("Attendance Summary")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total Days", total_days)
            metric_cols[1].metric("On Time", f"{on_time_days} ({(on_time_days/total_days*100):.1f}%)")
            metric_cols[2].metric("Late", f"{late_days} ({(late_days/total_days*100):.1f}%)")
            
            if late_days > 0:
                avg_late_mins = worker_all_records[worker_all_records['status'] == 'Late']['lateness_minutes'].mean()
                metric_cols[3].metric("Avg. Lateness", f"{avg_late_mins:.1f} min")
            
            # Show attendance by day of week
            st.subheader("Attendance Pattern by Day of Week")
            worker_all_records['day_of_week'] = worker_all_records['date'].dt.day_name()
            
            # Count by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = worker_all_records.groupby(['day_of_week', 'status']).size().unstack(fill_value=0)
            
            # Reindex to ensure all days are shown in correct order
            day_counts = day_counts.reindex(day_order)
            
            # Display as a bar chart
            st.bar_chart(day_counts)
            
            # Calculate attendance rate by day of week
            day_stats = worker_all_records.groupby('day_of_week').agg({
                'status': lambda x: (x == 'On Time').mean() * 100,
                'lateness_minutes': lambda x: x[x > 0].mean() if any(x > 0) else 0
            }).reset_index()
            
            day_stats.columns = ['Day of Week', 'On-Time Rate (%)', 'Avg. Lateness (min)']
            day_stats['Day of Week'] = pd.Categorical(day_stats['Day of Week'], categories=day_order, ordered=True)
            day_stats = day_stats.sort_values('Day of Week')
            
            # Show the day of week statistics
            st.dataframe(day_stats, use_container_width=True)
            
        elif worker_view == "Attendance Calendar":
            # Interactive calendar view for the worker
            st.subheader("Monthly Attendance Calendar")
            
            # Get all months with attendance data
            worker_all_records['year_month'] = worker_all_records['date'].dt.strftime('%Y-%m')
            months_with_data = sorted(worker_all_records['year_month'].unique())
            
            if not months_with_data:
                st.info("No attendance data available for calendar view.")
            else:
                # Month selector
                selected_year_month = st.selectbox("Select Month", months_with_data, index=len(months_with_data)-1)
                selected_year, selected_month = selected_year_month.split('-')
                selected_year, selected_month = int(selected_year), int(selected_month)
                
                # Create calendar data for the selected month
                cal = calendar.monthcalendar(selected_year, selected_month)
                month_df = worker_all_records[
                    (worker_all_records['date'].dt.year == selected_year) & 
                    (worker_all_records['date'].dt.month == selected_month)
                ].copy()
                
                # Get day-level data
                month_df['day'] = month_df['date'].dt.day
                month_records = month_df.set_index('day').to_dict('index')
                
                # Display calendar
                st.write(f"### {calendar.month_name[selected_month]} {selected_year}")
                
                # Create calendar table with detailed information
                cal_data = []
                for week in cal:
                    week_data = []
                    for day in week:
                        if day == 0:
                            # Empty cell
                            week_data.append("")
                        else:
                            # Check if there's data for this day
                            if day in month_records:
                                record = month_records[day]
                                status = record['status']
                                clock_in = record['clock_in_time']
                                expected = record.get('expected_time', 'N/A')
                                lateness = record.get('lateness_minutes', 0)
                                
                                # Format cell with attendance info
                                if status == 'On Time':
                                    cell_content = f"{day}: ✅ On Time\n{clock_in}"
                                else:
                                    cell_content = f"{day}: ⚠️ Late ({lateness} min)\n{clock_in}"
                            else:
                                cell_content = str(day)
                            
                            week_data.append(cell_content)
                    cal_data.append(week_data)
                
                # Display as a table
                cal_df = pd.DataFrame(cal_data, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                st.table(cal_df)
                
                # Show legend
                st.write("**Legend:** ✅ = On Time, ⚠️ = Late")
                
                # Show stats for the selected month
                st.subheader(f"Stats for {calendar.month_name[selected_month]} {selected_year}")
                month_total = len(month_df)
                month_on_time = len(month_df[month_df['status'] == 'On Time'])
                month_late = len(month_df[month_df['status'] == 'Late'])
                
                m_cols = st.columns(3)
                m_cols[0].metric("Total Days", month_total)
                m_cols[1].metric("On Time", f"{month_on_time} ({(month_on_time/month_total*100):.1f}%)")
                m_cols[2].metric("Late", f"{month_late} ({(month_late/month_total*100):.1f}%)")
                
                if month_late > 0:
                    avg_late = month_df[month_df['status'] == 'Late']['lateness_minutes'].mean()
                    st.metric("Average Lateness", f"{avg_late:.1f} minutes")
                
        elif worker_view == "Trends & Patterns":
            # Show trend over time
            st.subheader("Attendance Trend Over Time")
            
            # Sort by date
            worker_all_records = worker_all_records.sort_values('date')
            
            # Create a trend dataframe
            trend_df = worker_all_records[['date', 'status', 'lateness_minutes']].copy()
            trend_df['is_on_time'] = trend_df['status'] == 'On Time'
            trend_df['is_on_time_numeric'] = trend_df['is_on_time'].astype(int)
            
            # Calculate rolling average (7-day window)
            if len(trend_df) >= 7:
                trend_df['on_time_rate_7day'] = trend_df['is_on_time_numeric'].rolling(window=7).mean() * 100
                
                # Show the 7-day rolling average trend
                st.subheader("7-Day On-Time Rate (%)")
                rolling_chart_data = trend_df.dropna(subset=['on_time_rate_7day']).set_index('date')[['on_time_rate_7day']]
                st.line_chart(rolling_chart_data, use_container_width=True)
            
            # Trend chart for on-time vs. late
            st.subheader("Daily Attendance (1 = On Time, 0 = Late)")
            st.line_chart(trend_df.set_index('date')['is_on_time_numeric'], use_container_width=True)
            
            # Display lateness trend when late
            late_records = trend_df[trend_df['status'] == 'Late']
            if not late_records.empty:
                st.subheader("Lateness Minutes Trend (when late)")
                st.line_chart(late_records.set_index('date')['lateness_minutes'], use_container_width=True)
                
                # Calculate and display lateness distribution
                st.subheader("Lateness Distribution (minutes)")
                late_dist = late_records['lateness_minutes'].value_counts().sort_index()
                st.bar_chart(late_dist)
                
                # Calculate patterns
                if len(worker_all_records) >= 5:
                    st.subheader("Attendance Patterns")
                    
                    # Check for day of week patterns
                    worker_all_records['day_of_week'] = worker_all_records['date'].dt.day_name()
                    day_late_rate = worker_all_records.groupby('day_of_week').apply(
                        lambda x: (x['status'] == 'Late').mean() * 100
                    ).sort_values(ascending=False)
                    
                    # Find the day with highest lateness rate
                    worst_day = day_late_rate.index[0] 
                    worst_day_rate = day_late_rate.iloc[0]
                    
                    # Find the day with lowest lateness rate
                    best_day = day_late_rate.index[-1]
                    best_day_rate = day_late_rate.iloc[-1]
                    
                    st.info(f"Most punctual day: **{best_day}** (Late rate: {best_day_rate:.1f}%)")
                    st.warning(f"Least punctual day: **{worst_day}** (Late rate: {worst_day_rate:.1f}%)")
                    
                    # Check consecutive patterns
                    consecutive_lates = 0
                    max_consecutive_lates = 0
                    
                    for status in worker_all_records.sort_values('date')['status']:
                        if status == 'Late':
                            consecutive_lates += 1
                            max_consecutive_lates = max(max_consecutive_lates, consecutive_lates)
                        else:
                            consecutive_lates = 0
                    
                    if max_consecutive_lates >= 2:
                        st.warning(f"Longest streak of consecutive late arrivals: **{max_consecutive_lates}** days")
                    
                    # Check for improvement
                    if len(worker_all_records) >= 10:
                        # Compare first half vs second half
                        half_point = len(worker_all_records) // 2
                        first_half = worker_all_records.iloc[:half_point]
                        second_half = worker_all_records.iloc[half_point:]
                        
                        first_half_late_rate = (first_half['status'] == 'Late').mean() * 100
                        second_half_late_rate = (second_half['status'] == 'Late').mean() * 100
                        
                        if second_half_late_rate < first_half_late_rate:
                            improvement = first_half_late_rate - second_half_late_rate
                            st.success(f"Showing improvement: Late rate decreased by **{improvement:.1f}%** over time")
                        elif second_half_late_rate > first_half_late_rate:
                            decline = second_half_late_rate - first_half_late_rate
                            st.error(f"Punctuality declining: Late rate increased by **{decline:.1f}%** over time")
            
        else:  # Monthly Analysis
            # Monthly attendance summary
            st.subheader("Monthly Attendance Summary")
            worker_all_records['year_month'] = worker_all_records['date'].dt.strftime('%Y-%m')
            monthly_summary = worker_all_records.groupby('year_month').agg({
                'worker_id': 'count',
                'status': lambda x: (x == 'On Time').sum(),
                'lateness_minutes': 'sum'
            }).reset_index()
            
            monthly_summary.columns = ['Month', 'Days Worked', 'Days On Time', 'Total Late Minutes']
            monthly_summary['Days Late'] = monthly_summary['Days Worked'] - monthly_summary['Days On Time']
            monthly_summary['On Time Rate (%)'] = (monthly_summary['Days On Time'] / monthly_summary['Days Worked'] * 100).round(1)
            monthly_summary['Avg. Lateness (min)'] = (monthly_summary['Total Late Minutes'] / monthly_summary['Days Late']).fillna(0).round(1)
            
            # Show monthly summary table
            st.dataframe(monthly_summary.sort_values('Month', ascending=False), use_container_width=True)
            
            # Visualize monthly trends
            if len(monthly_summary) > 1:
                st.subheader("Monthly On-Time Rate Trend")
                monthly_chart_data = monthly_summary[['Month', 'On Time Rate (%)']]
                monthly_chart_data = monthly_chart_data.sort_values('Month')
                st.line_chart(monthly_chart_data.set_index('Month'), use_container_width=True)
            
            # Recent attendance
            st.subheader("Recent Attendance")
            recent_records = worker_all_records.sort_values('date', ascending=False).head(10)
            st.dataframe(recent_records[['date', 'clock_in_time', 'expected_time', 'status', 'lateness_minutes']], use_container_width=True)
        
        # Generate detailed worker report
        st.subheader("Generate Report")
        report_type = st.radio(
            "Report Type",
            ["Comprehensive", "Attendance Only", "Lateness Analysis"],
            horizontal=True
        )
        
        if st.button("Generate Detailed Report"):
            # Create a more detailed dataframe for the report
            report_df = worker_all_records.copy()
            
            # Add day of week
            report_df['day_of_week'] = report_df['date'].dt.day_name()
            
            # Format date for better readability
            report_df['formatted_date'] = report_df['date'].dt.strftime('%Y-%m-%d')
            
            # Select and order columns for the report based on type
            if report_type == "Comprehensive":
                report_columns = [
                    'formatted_date', 'day_of_week', 'clock_in_time', 
                    'expected_time', 'status', 'lateness_minutes', 
                    'department', 'position', 'shift_type'
                ]
            elif report_type == "Attendance Only":
                report_columns = [
                    'formatted_date', 'day_of_week', 'clock_in_time', 
                    'expected_time', 'status'
                ]
            else:  # Lateness Analysis
                report_df = report_df[report_df['status'] == 'Late']
                report_columns = [
                    'formatted_date', 'day_of_week', 'clock_in_time', 
                    'expected_time', 'lateness_minutes'
                ]
            
            # Add summary statistics to the report
            report_buf = io.StringIO()
            
            # Write header and worker info
            report_buf.write(f"Attendance Report for: {selected_worker}\n")
            report_buf.write(f"Department: {report_df['department'].iloc[0] if not report_df.empty else 'N/A'}\n")
            report_buf.write(f"Position: {report_df['position'].iloc[0] if not report_df.empty else 'N/A'}\n")
            report_buf.write(f"Shift Type: {report_df['shift_type'].iloc[0] if not report_df.empty else 'N/A'}\n")
            report_buf.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary statistics
            report_buf.write("SUMMARY STATISTICS\n")
            report_buf.write(f"Total Days Recorded: {total_days}\n")
            report_buf.write(f"Days On Time: {on_time_days} ({(on_time_days/total_days*100):.1f}%)\n")
            report_buf.write(f"Days Late: {late_days} ({(late_days/total_days*100):.1f}%)\n")
            
            if late_days > 0:
                report_buf.write(f"Average Lateness: {avg_late_mins:.1f} minutes\n")
                
                # Calculate streak information
                current_streak = 0
                max_on_time_streak = 0
                on_time_streak = 0
                late_streak = 0
                max_late_streak = 0
                
                for status in trend_df['status']:
                    if status == 'On Time':
                        on_time_streak += 1
                        late_streak = 0
                        if on_time_streak > max_on_time_streak:
                            max_on_time_streak = on_time_streak
                    else:  # Late
                        late_streak += 1
                        on_time_streak = 0
                        if late_streak > max_late_streak:
                            max_late_streak = late_streak
                
                report_buf.write(f"Longest On-Time Streak: {max_on_time_streak} days\n")
                report_buf.write(f"Longest Late Streak: {max_late_streak} days\n\n")
            
            # Write monthly breakdown
            monthly_data = report_df.groupby(report_df['date'].dt.strftime('%Y-%m')).agg({
                'worker_id': 'count',
                'status': lambda x: (x == 'On Time').sum()
            })
            
            report_buf.write("MONTHLY BREAKDOWN\n")
            for month, row in monthly_data.iterrows():
                total = row['worker_id']
                on_time = row['status']
                report_buf.write(f"{month}: {on_time}/{total} days on time ({(on_time/total*100):.1f}%)\n")
            
            report_buf.write("\nDETAILED RECORDS\n")
            
            # Write the CSV data
            csv_data = report_df[report_columns].sort_values('formatted_date', ascending=False).to_csv(index=False)
            report_buf.write(csv_data)
            
            # Get the full report as a string
            full_report = report_buf.getvalue()
            
            # Create CSV data for download
            report_csv = report_df[report_columns].sort_values('formatted_date', ascending=False).to_csv(index=False)
            
            # Offer downloads - both CSV and full report
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download CSV Report",
                    data=report_csv,
                    file_name=f"{selected_worker}_attendance_report.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="📥 Download Full Report (TXT)",
                    data=full_report,
                    file_name=f"{selected_worker}_full_attendance_report.txt",
                    mime="text/plain"
                )
    
    # Visualizations
    st.subheader("Attendance Visualizations")
    
    # Attendance by date
    if time_period == "Weekly":
        attendance_by_date = filtered_df.groupby(['date', 'status']).size().unstack(fill_value=0)
        st.bar_chart(attendance_by_date)
        
        # Weekly statistics table
        st.subheader("Weekly Summary")
        days_of_week = filtered_df.copy()
        days_of_week['day_of_week'] = days_of_week['date'].dt.day_name()
        day_stats = days_of_week.groupby('day_of_week').agg({
            'worker_id': 'count',
            'status': lambda x: (x == 'On Time').sum(),
            'lateness_minutes': 'mean'
        }).reset_index()
        day_stats.columns = ['Day', 'Total Attendance', 'On Time Count', 'Avg Lateness (min)']
        day_stats['Late Count'] = day_stats['Total Attendance'] - day_stats['On Time Count']
        day_stats['On Time %'] = (day_stats['On Time Count'] / day_stats['Total Attendance'] * 100).round(1)
        
        # Sort by days of week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats['Day'] = pd.Categorical(day_stats['Day'], categories=days_order, ordered=True)
        day_stats = day_stats.sort_values('Day')
        
        st.dataframe(day_stats, use_container_width=True)
    else:
        # Monthly calendar view
        st.subheader(f"Monthly Attendance: {calendar.month_name[selected_month]} {selected_year}")
        
        # Create calendar data
        cal = calendar.monthcalendar(selected_year, selected_month)
        month_df = filtered_df.copy()
        month_df['day'] = month_df['date'].dt.day
        
        # Count attendance per day
        daily_counts = month_df.groupby('day').size().to_dict()
        daily_late = month_df[month_df['status'] == 'Late'].groupby('day').size().to_dict()
        
        # Display calendar
        cal_data = []
        for week in cal:
            week_data = []
            for day in week:
                if day == 0:
                    # Empty cell
                    week_data.append("")
                else:
                    # Day cell with attendance data
                    count = daily_counts.get(day, 0)
                    late = daily_late.get(day, 0)
                    week_data.append(f"{day}: {count} in ({late} late)" if count > 0 else str(day))
            cal_data.append(week_data)
        
        # Display as a table
        cal_df = pd.DataFrame(cal_data, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        st.table(cal_df)
        
        # Monthly trends
        daily_status = month_df.groupby(['day', 'status']).size().unstack(fill_value=0)
        if not daily_status.empty:
            st.line_chart(daily_status)
    
    # Attendance by worker
    attendance_by_worker = filtered_df.groupby(['name', 'status']).size().unstack(fill_value=0)
    
    if not attendance_by_worker.empty:
        st.subheader("Attendance by Worker")
        st.bar_chart(attendance_by_worker)
    
    # Detailed worker statistics
    st.subheader("Worker Details")
    
    # Group by worker
    worker_stats = filtered_df.groupby('name').agg({
        'status': lambda x: (x == 'On Time').sum(),
        'lateness_minutes': 'sum',
        'worker_id': 'first',
        'department': 'first'
    }).reset_index()
    
    worker_stats.columns = ['Name', 'On Time Count', 'Total Lateness (min)', 'Worker ID', 'Department']
    worker_stats['Attendance Count'] = filtered_df.groupby('name').size().values
    worker_stats['Late Count'] = worker_stats['Attendance Count'] - worker_stats['On Time Count']
    worker_stats['On Time %'] = (worker_stats['On Time Count'] / worker_stats['Attendance Count'] * 100).round(1)
    
    # Reorder columns
    worker_stats = worker_stats[['Name', 'Worker ID', 'Department', 'Attendance Count', 
                                 'On Time Count', 'Late Count', 'On Time %', 
                                 'Total Lateness (min)']]
    
    st.dataframe(worker_stats, use_container_width=True)
    
    # Download detailed statistics
    csv = worker_stats.to_csv(index=False)
    time_label = f"Week_{selected_week}" if time_period == "Weekly" else f"{calendar.month_name[selected_month]}_{selected_year}"
    
    st.download_button(
        label="📥 Download Statistics CSV",
        data=csv,
        file_name=f"attendance_stats_{time_label}.csv",
        mime="text/csv"
    )

def main():
    """Main application"""
    st.set_page_config(
        page_title="Real-time Worker Tracking System",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Real-time Worker Tracking System")
    
    # Initialize database and load data on startup
    init_database()
    # Remove old CSV and pickle files
    remove_old_files()
    # Load data from database
    load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🎥 Real-time Clock In", "➕ Add Worker", "📊 Attendance Records", "👥 Workers Management", "📈 Attendance Statistics"]
    )
    
    # Display selected page
    if page == "➕ Add Worker":
        add_worker_page()
    elif page == "🎥 Real-time Clock In":
        realtime_clock_in_page()
    elif page == "📊 Attendance Records":
        attendance_records_page()
    elif page == "👥 Workers Management":
        workers_management_page()
    elif page == "📈 Attendance Statistics":
        attendance_statistics_page()
    
    # Display current stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 System Stats")
    st.sidebar.metric("Registered Workers", len(st.session_state.workers_db))
    st.sidebar.metric("Total Clock-ins", len(st.session_state.attendance_records))
    
    # Today's attendance
    today = date.today().strftime("%Y-%m-%d")
    today_attendance = len([r for r in st.session_state.attendance_records if r['date'] == today])
    st.sidebar.metric("Today's Attendance", today_attendance)
    
    # Camera status
    if st.session_state.camera_active:
        st.sidebar.success("📹 Camera Active")
    else:
        st.sidebar.info("📹 Camera Inactive")
    
    # Quick actions in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Quick Actions")
    if st.sidebar.button("🔄 Refresh Data"):
        load_data()
        st.sidebar.success("Data refreshed!")
    
    if st.sidebar.button("💾 Save Data"):
        save_data()
        st.sidebar.success("Data saved!")
    
    # Cleanup on app shutdown
    if not st.session_state.get('camera_active', False):
        if 'video_processor' in st.session_state:
            st.session_state.video_processor.stop_camera()

if __name__ == "__main__":
    main()