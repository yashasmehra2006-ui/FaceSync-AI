import cv2
import numpy as np
import base64
import sqlite3
import time
import io
from datetime import datetime
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import face_recognition

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# =====================================================================
# TIMETABLE DEFINITION (CSE Panel - A)
# =====================================================================
TIMETABLE_PANEL_A = {
    "Monday": [
        ("10:45", "11:45", "SEM"),
        ("11:45", "12:45", "PE-I (Cloud Computing)"),
        ("13:30", "15:30", "PBL-III / CN Lab"),
        ("15:45", "16:45", "PBL-III (Elective)")
    ],
    "Tuesday": [
        ("09:30", "10:30", "SEM"),
        ("10:45", "11:45", "AIES"),
        ("11:45", "12:45", "Computer Networks"),
        ("13:30", "15:30", "IoT Lab / PBL-III")
    ],
    "Wednesday": [
        ("10:45", "11:45", "AIES"),
        ("11:45", "12:45", "PE-I (Cloud Computing)"),
        ("13:30", "15:30", "PE-I Lab / AIES Lab")
    ],
    "Thursday": [
        ("09:30", "10:30", "Mentor-Mentee"),
        ("10:45", "11:45", "Computer Networks"),
        ("11:45", "12:45", "AIES"),
        ("13:30", "15:30", "AIES Lab / IoT Lab")
    ],
    "Friday": [
        ("08:50", "09:30", "PE-I (Cloud Computing)"),
        ("09:30", "10:30", "SEM"),
        ("10:45", "11:45", "MCPT"),
        ("11:45", "12:45", "MCPT"),
        ("13:30", "15:30", "CN Lab / PE-I Lab")
    ]
}

def get_current_subject():
    now = datetime.now()
    day_name = now.strftime("%A")
    if day_name not in TIMETABLE_PANEL_A:
        return "General / Off-Hours"
    
    current_time_minutes = int(now.strftime("%H")) * 60 + int(now.strftime("%M"))

    for start_str, end_str, subject in TIMETABLE_PANEL_A[day_name]:
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        if (sh * 60 + sm) <= current_time_minutes <= (eh * 60 + em):
            return subject
            
    return "General / Off-Hours"

# =====================================================================
# CORE LOGIC CLASSES
# =====================================================================
class ImagePreprocessor:
    @staticmethod
    def to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def downscale(frame: np.ndarray, scale: float = 0.20) -> np.ndarray:
        return cv2.resize(frame, (0, 0), fx=scale, fy=scale)

class MetricVectorMatcher:
    def __init__(self, tolerance: float = 0.50):
        self.tolerance = tolerance

    def find_best_match(self, candidate_encoding, known_encodings, known_prns, known_names):
        if not known_encodings:
            return None, "Unknown", 1.0
        distances = face_recognition.face_distance(known_encodings, candidate_encoding)
        best_idx = int(np.argmin(distances))
        min_distance = distances[best_idx]
        if min_distance <= self.tolerance:
            return known_prns[best_idx], known_names[best_idx], min_distance
        return None, "Unknown", min_distance

class TimetableAntiProxyController:
    def __init__(self):
        self.logged_slots = {}

    def is_eligible(self, prn: str, subject: str, date_str: str) -> bool:
        key = (prn, date_str, subject)
        if key in self.logged_slots:
            return False
        self.logged_slots[key] = True
        return True

class AttendanceDatabase:
    def __init__(self, db_path: str = "facesync.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    prn TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    encoding BLOB NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prn TEXT NOT NULL,
                    name TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    FOREIGN KEY (prn) REFERENCES students (prn)
                )
            """)
            conn.commit()

    def fetch_all_student_vectors(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prn, name, encoding FROM students")
            rows = cursor.fetchall()
        prns, names, encodings = [], [], []
        for prn, name, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float64)
            prns.append(prn)
            names.append(name)
            encodings.append(arr)
        return prns, names, encodings

    def record_attendance(self, prn: str, name: str, subject: str):
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO attendance_logs (prn, name, subject, timestamp, date) VALUES (?, ?, ?, ?, ?)",
                           (prn, name, subject, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")))
            conn.commit()

    def register_student(self, prn: str, name: str, department: str, encoding: np.ndarray):
        blob = encoding.tobytes()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO students (prn, name, department, encoding) 
                VALUES (?, ?, ?, ?)
            """, (prn, name, department, blob))
            conn.commit()

    def fetch_recent_logs(self, limit: int = 15):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prn, name, subject, timestamp, date 
                FROM attendance_logs 
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        logs = []
        for prn, name, subject, timestamp, date in rows:
            logs.append({"prn": prn, "name": name, "subject": subject, "timestamp": timestamp, "date": date})
        return logs

    def fetch_all_logs_csv(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prn, name, subject, timestamp, date FROM attendance_logs ORDER BY id DESC")
            rows = cursor.fetchall()
        
        output = io.StringIO()
        output.write("Log ID,PRN,Student Name,Subject,Timestamp,Date\n")
        for row in rows:
            output.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
        
        return output.getvalue()

db = AttendanceDatabase()
matcher = MetricVectorMatcher()
anti_proxy = TimetableAntiProxyController()
preprocessor = ImagePreprocessor()

class RegisterPayload(BaseModel):
    prn: str
    name: str
    department: str
    images: List[str]

@app.get("/api/logs")
async def get_logs():
    return db.fetch_recent_logs()

@app.get("/api/export/csv")
async def export_csv():
    csv_data = db.fetch_all_logs_csv()
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=panel_a_attendance.csv"}
    )

@app.post("/api/register")
async def register_student_api(payload: RegisterPayload):
    try:
        collected_encodings = []
        for img_data in payload.images:
            header, encoded = img_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb_frame)
            if encodings:
                collected_encodings.append(encodings[0])

        if not collected_encodings:
            return {"success": False, "message": "No face detected across burst frames. Try again."}
        
        mean_encoding = np.mean(collected_encodings, axis=0)
        db.register_student(payload.prn, payload.name, payload.department, mean_encoding)
        
        return {"success": True, "message": f"Successfully registered {payload.name}!"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()
    prns, names, encodings = db.fetch_all_student_vectors()
    
    last_results = []
    frame_count = 0

    try:
        while True:
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                frame_count += 1
                current_subject = get_current_subject()
                current_date = datetime.now().strftime("%Y-%m-%d")

                if frame_count % 2 == 0:
                    rgb_frame = preprocessor.to_rgb(bgr_frame)
                    small_rgb = preprocessor.downscale(rgb_frame, scale=0.20)

                    face_locations = face_recognition.face_locations(small_rgb)
                    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

                    last_results = []

                    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                        prn, name, dist = matcher.find_best_match(encoding, encodings, prns, names)

                        box_color = (0, 255, 0) if prn else (255, 0, 0)
                        if prn and anti_proxy.is_eligible(prn, current_subject, current_date):
                            db.record_attendance(prn, name, current_subject)

                        last_results.append(((top, right, bottom, left), name, dist, box_color))

                for (top, right, bottom, left), name, dist, box_color in last_results:
                    cv2.rectangle(bgr_frame, (left, top), (right, bottom), box_color, 2)
                    label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
                    cv2.putText(bgr_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                _, buffer = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                processed_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(f"data:image/jpeg;base64,{processed_base64}")

    except WebSocketDisconnect:
        print("Client disconnected from video stream.")
