
import cv2
import os
import sqlite3
import winsound
import pandas as pd
import tkinter as tk
import numpy as np
import shutil

from tkinter import ttk, messagebox, simpledialog
from datetime import datetime, date
from PIL import Image, ImageTk


# PROJECT SETTINGS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

THRESHOLD = 80
SAMPLES_REQUIRED = 30
FACE_SIZE = (200, 200)


# DASHBOARD COLORS

BG = "#F3F4F8"
SIDEBAR = "#171B35"
PURPLE = "#6D5AE6"
GREEN = "#16A36A"
ORANGE = "#E99A32"
RED = "#DF5360"
BLUE = "#3182CE"
TEXT = "#252944"
MUTED = "#7C8198"
WHITE = "#FFFFFF"

# CAMERA SOUND

def play_camera_sound():
    try:
        winsound.Beep(1500, 120)
    except Exception as exc:
        print("Could not play camera sound:", exc)

# DATABASE

def connect_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def get_table_columns(table_name):
    with connect_db() as conn:
        return [
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        ]


def setup_database():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll TEXT NOT NULL,
                name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL
            )
        """)

        conn.commit()

        indexes = conn.execute(
            "PRAGMA index_list(attendance)"
        ).fetchall()

        needs_migration = False

        for index in indexes:
            if index[2]:
                columns = conn.execute(
                    f'PRAGMA index_info("{index[1]}")'
                ).fetchall()

                names = {column[2] for column in columns}

                if names == {"roll", "attendance_date"}:
                    needs_migration = True
                    break

        if needs_migration:
            conn.execute("""
                CREATE TABLE attendance_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    roll TEXT NOT NULL,
                    name TEXT NOT NULL,
                    attendance_date TEXT NOT NULL,
                    attendance_time TEXT NOT NULL
                )
            """)

            conn.execute("""
                INSERT INTO attendance_new (
                    id, roll, name,
                    attendance_date, attendance_time
                )
                SELECT
                    id, roll, name,
                    attendance_date, attendance_time
                FROM attendance
            """)

            conn.execute("DROP TABLE attendance")
            conn.execute("""
                ALTER TABLE attendance_new
                RENAME TO attendance
            """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_roll_date
            ON attendance (roll, attendance_date)
        """)

        conn.commit()

    print("Database in use:", DB_PATH)


def get_students():
    with connect_db() as conn:
        rows = conn.execute("""
            SELECT rowid, roll, name
            FROM students
            ORDER BY roll
        """).fetchall()

    return {
        row[0]: {"roll": row[1], "name": row[2]}
        for row in rows
    }


def mark_attendance(student):
    now = datetime.now()
    roll = student["roll"].strip()
    name = student["name"].strip()

    attendance_date = now.strftime("%Y-%m-%d")
    attendance_time = now.strftime("%H:%M:%S")

    try:
        with connect_db() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT 1 FROM students
                WHERE TRIM(roll) = ?
                LIMIT 1
            """, (roll,))

            if cur.fetchone() is None:
                return "error", "Student not found in database."

            cur.execute("""
                INSERT INTO attendance (
                    roll, name,
                    attendance_date, attendance_time
                )
                VALUES (?, ?, ?, ?)
            """, (
                roll, name, attendance_date, attendance_time
            ))

            conn.commit()

        return "success", attendance_time

    except sqlite3.Error as exc:
        print("Attendance database error:", exc)
        return "error", str(exc)

# FACE DETECTOR AND RECOGNIZER

def create_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "LBPH recognizer is unavailable.\n\n"
            "Run this command in the VS Code terminal:\n"
            "python -m pip install opencv-contrib-python"
        )

    return cv2.face.LBPHFaceRecognizer_create()


def load_face_detector():
    filename = "haarcascade_frontalface_default.xml"
    candidates = []

    cascade_data = getattr(cv2, "data", None)
    cascade_dir = getattr(cascade_data, "haarcascades", None)

    if cascade_dir:
        candidates.append(os.path.join(cascade_dir, filename))

    candidates.append(
        os.path.join(BASE_DIR, "haarcascades", filename)
    )

    for path in candidates:
        if os.path.isfile(path):
            detector = cv2.CascadeClassifier(path)

            if not detector.empty():
                return detector

    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        "Install OpenCV using:\n"
        "python -m pip install opencv-contrib-python"
    )
    
# TRAIN FACE MODEL

def train_model():
    students = get_students()

    if not students:
        return False

    images = []
    labels = []

    for student_id in students:
        folder = os.path.join(DATASET_DIR, str(student_id))

        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            path = os.path.join(folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            images.append(cv2.resize(image, FACE_SIZE))
            labels.append(student_id)

    if not images:
        return False

    recognizer = create_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(MODEL_PATH)

    return True

# MAIN DASHBOARD APPLICATION

class AttendanceApp:

    def __init__(self, root):
        self.root = root
        self.root.title("FaceTrack | Attendance Management System")
        self.root.geometry("1200x780")
        self.root.minsize(980, 650)
        self.root.configure(bg=BG)

        self.camera = None
        self.running = False
        self.recognizer = None
        self.detector = load_face_detector()

        self.students = {}
        self.last_recognized_id = None
        self.recognition_count = 0

        self.status_text = tk.StringVar(
            value="Welcome! Register students to get started."
        )

        self.total_students_text = tk.StringVar(value="0")
        self.present_text = tk.StringVar(value="0")
        self.absent_text = tk.StringVar(value="0")
        self.rate_text = tk.StringVar(value="0%")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Dashboard.Treeview",
            background=WHITE,
            foreground=TEXT,
            fieldbackground=WHITE,
            rowheight=32,
            font=("Segoe UI", 10)
        )

        style.configure(
            "Dashboard.Treeview.Heading",
            background="#EAE8FF",
            foreground=TEXT,
            font=("Segoe UI", 10, "bold")
        )

        # MAIN LAYOUT
        self.sidebar = tk.Frame(
            root, bg=SIDEBAR, width=230
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.content = tk.Frame(root, bg=BG)
        self.content.pack(
            side="left", fill="both", expand=True
        )

        # SIDEBAR BRANDING
        tk.Label(
            self.sidebar,
            text="◈ FaceTrack",
            font=("Segoe UI", 22, "bold"),
            bg=SIDEBAR,
            fg=WHITE
        ).pack(anchor="w", padx=20, pady=(28, 4))

        tk.Label(
            self.sidebar,
            text="ATTENDANCE SYSTEM",
            font=("Segoe UI", 8, "bold"),
            bg=SIDEBAR,
            fg="#A6ACCE"
        ).pack(anchor="w", padx=23, pady=(0, 28))

        self.add_nav("▦  Dashboard", self.show_dashboard)
        self.add_nav("＋  Register Student", self.register_student)
        self.add_nav("✕  Delete Student", self.delete_student)
        self.add_nav("◎  Train Face Model", self.train_faces)
        self.add_nav("▶  Start Camera", self.start_camera)
        self.add_nav("■  Stop Camera", self.stop_camera)
        self.add_nav("♙  View Students", self.view_students)
        self.add_nav("▤  Attendance Records", self.view_attendance)
        self.add_nav("⇩  Export to Excel", self.export_excel)
        self.add_nav("▱  Open Reports", self.open_reports_folder)

        tk.Label(
            self.sidebar,
            text="POWERED BY PYTHON + OPENCV",
            font=("Segoe UI", 8),
            bg=SIDEBAR,
            fg="#A6ACCE"
        ).pack(side="bottom", pady=20)

        # DASHBOARD HEADER
        header = tk.Frame(self.content, bg=BG)
        header.pack(fill="x", padx=26, pady=(22, 8))

        tk.Label(
            header,
            text="Dashboard Overview",
            font=("Segoe UI", 24, "bold"),
            bg=BG,
            fg=TEXT
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Manage students and monitor attendance in real time.",
            font=("Segoe UI", 10),
            bg=BG,
            fg=MUTED
        ).pack(anchor="w", pady=(4, 0))

        # STATISTICS CARDS
        stats = tk.Frame(self.content, bg=BG)
        stats.pack(fill="x", padx=18, pady=10)

        self.stat_card(
            stats, "TOTAL STUDENTS",
            self.total_students_text, PURPLE, "♙", 0
        )

        self.stat_card(
            stats, "PRESENT TODAY",
            self.present_text, GREEN, "✓", 1
        )

        self.stat_card(
            stats, "ABSENT TODAY",
            self.absent_text, ORANGE, "○", 2
        )

        self.stat_card(
            stats, "ATTENDANCE RATE",
            self.rate_text, BLUE, "%", 3
        )

        # CAMERA CARD
        camera_card = tk.Frame(
            self.content,
            bg=WHITE,
            highlightbackground="#E1E4EF",
            highlightthickness=1
        )
        camera_card.pack(
            fill="both", expand=True,
            padx=26, pady=(8, 10)
        )

        camera_header = tk.Frame(camera_card, bg=WHITE)
        camera_header.pack(fill="x", padx=18, pady=13)

        tk.Label(
            camera_header,
            text="Live Face Recognition",
            font=("Segoe UI", 15, "bold"),
            bg=WHITE,
            fg=TEXT
        ).pack(side="left")

        tk.Label(
            camera_header,
            text="BUILT-IN LAPTOP CAMERA",
            font=("Segoe UI", 9, "bold"),
            bg="#E5F7EF",
            fg="#138353",
            padx=10,
            pady=5
        ).pack(side="right")

        self.video_label = tk.Label(
            camera_card,
            text="Camera preview will appear here",
            bg=SIDEBAR,
            fg="#C7CBE0",
            font=("Segoe UI", 13),
            anchor="center"
        )
        self.video_label.pack(
            fill="both", expand=True,
            padx=18, pady=(0, 12)
        )

        controls = tk.Frame(camera_card, bg=WHITE)
        controls.pack(fill="x", padx=18, pady=(0, 12))

        self.make_button(
            controls, "▶  Start Camera",
            self.start_camera, PURPLE
        ).pack(side="left", padx=(0, 8))

        self.make_button(
            controls, "■  Stop Camera",
            self.stop_camera, "#4B5563"
        ).pack(side="left")

        # STATUS BAR
        tk.Label(
            self.content,
            textvariable=self.status_text,
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=900
        ).pack(fill="x", padx=28, pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        self.update_student_count()
        self.update_dashboard_stats()
        
    # DASHBOARD UI HELPERS

    def add_nav(self, text, command):
        button = tk.Button(
            self.sidebar,
            text=text,
            command=command,
            bg=SIDEBAR,
            fg="#E1E4F5",
            activebackground="#34305C",
            activeforeground=WHITE,
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
            anchor="w",
            padx=20,
            pady=11,
            cursor="hand2"
        )
        button.pack(fill="x", padx=8, pady=2)

        button.bind(
            "<Enter>",
            lambda event: button.configure(bg="#292D50")
        )
        button.bind(
            "<Leave>",
            lambda event: button.configure(bg=SIDEBAR)
        )

    def stat_card(self, parent, title, variable, color, symbol, column):
        card = tk.Frame(
            parent,
            bg=WHITE,
            highlightbackground="#E1E4EF",
            highlightthickness=1
        )

        card.grid(
            row=0, column=column,
            padx=6, sticky="nsew"
        )

        parent.columnconfigure(column, weight=1)

        tk.Label(
            card,
            text=symbol,
            font=("Segoe UI", 20, "bold"),
            fg=color,
            bg=WHITE
        ).pack(anchor="w", padx=14, pady=(10, 1))

        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 8, "bold"),
            fg=MUTED,
            bg=WHITE
        ).pack(anchor="w", padx=14)

        tk.Label(
            card,
            textvariable=variable,
            font=("Segoe UI", 23, "bold"),
            fg=TEXT,
            bg=WHITE
        ).pack(anchor="w", padx=14, pady=(2, 12))

    def make_button(self, parent, text, command, color):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=WHITE,
            activebackground=color,
            activeforeground=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2"
        )

    def show_dashboard(self):
        self.update_student_count()
        self.update_dashboard_stats()
        self.status_text.set("Dashboard refreshed.")

    # DASHBOARD STATISTICS

    def update_student_count(self):
        try:
            with connect_db() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM students"
                ).fetchone()[0]

            self.total_students_text.set(str(total))

        except sqlite3.Error as exc:
            self.status_text.set(
                f"Could not count students: {exc}"
            )

    def update_dashboard_stats(self):
        today = date.today().strftime("%Y-%m-%d")

        try:
            with connect_db() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM students"
                ).fetchone()[0]

                present = conn.execute("""
                    SELECT COUNT(DISTINCT s.roll)
                    FROM students AS s
                    JOIN attendance AS a
                      ON TRIM(a.roll) = TRIM(s.roll)
                    WHERE a.attendance_date = ?
                """, (today,)).fetchone()[0]

            absent = max(total - present, 0)
            rate = (present / total * 100) if total else 0

            self.total_students_text.set(str(total))
            self.present_text.set(str(present))
            self.absent_text.set(str(absent))
            self.rate_text.set(f"{rate:.1f}%")

        except sqlite3.Error as exc:
            self.status_text.set(
                f"Could not refresh dashboard: {exc}"
            )

    # REGISTER STUDENT

    def register_student(self):
        if self.running:
            messagebox.showwarning(
                "Camera Running",
                "Stop the attendance camera before registration.",
                parent=self.root
            )
            return

        name = simpledialog.askstring(
            "Student Registration",
            "Enter student name:",
            parent=self.root
        )

        if not name or not name.strip():
            return

        roll = simpledialog.askstring(
            "Student Registration",
            "Enter roll number:",
            parent=self.root
        )

        if not roll or not roll.strip():
            return

        name = name.strip()
        roll = roll.strip()

        try:
            with connect_db() as conn:
                cur = conn.cursor()

                existing = cur.execute("""
                    SELECT roll FROM students
                    WHERE TRIM(roll) = ?
                    LIMIT 1
                """, (roll,)).fetchone()

                if existing:
                    messagebox.showerror(
                        "Duplicate Roll Number",
                        f"Roll number '{roll}' is already registered.",
                        parent=self.root
                    )
                    return

                columns = [
                    row[1] for row in cur.execute(
                        "PRAGMA table_info(students)"
                    ).fetchall()
                ]

                if "created_at" in columns:
                    cur.execute("""
                        INSERT INTO students
                            (roll, name, created_at)
                        VALUES (?, ?, ?)
                    """, (
                        roll,
                        name,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
                else:
                    cur.execute("""
                        INSERT INTO students (roll, name)
                        VALUES (?, ?)
                    """, (roll, name))

                student_id = cur.lastrowid
                conn.commit()

        except sqlite3.Error as exc:
            messagebox.showerror(
                "Registration Error",
                str(exc),
                parent=self.root
            )
            return

        self.update_student_count()
        self.update_dashboard_stats()

        folder = os.path.join(DATASET_DIR, str(student_id))
        os.makedirs(folder, exist_ok=True)

        # Built-in laptop camera only.
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()

            with connect_db() as conn:
                conn.execute(
                    "DELETE FROM students WHERE rowid = ?",
                    (student_id,)
                )
                conn.commit()

            shutil.rmtree(folder, ignore_errors=True)
            self.update_student_count()
            self.update_dashboard_stats()

            messagebox.showerror(
                "Camera Error",
                "Could not open the laptop camera.",
                parent=self.root
            )
            return

        count = 0
        frame_count = 0

        messagebox.showinfo(
            "Capture Photos",
            f"Capturing {SAMPLES_REQUIRED} face samples for "
            f"{name} ({roll}).\n\n"
            "Look at the camera and slowly change your head angle.\n"
            "Press Q or Esc in the camera window to cancel.",
            parent=self.root
        )

        try:
            while count < SAMPLES_REQUIRED:
                ok, frame = camera.read()

                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                frame_count += 1

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2
                    )

                    if frame_count % 3 == 0:
                        face = cv2.resize(
                            gray[y:y + h, x:x + w],
                            FACE_SIZE
                        )

                        image_path = os.path.join(
                            folder, f"{count:03d}.jpg"
                        )

                        if cv2.imwrite(image_path, face):
                            count += 1

                    # Capture one face at a time.
                    break

                cv2.putText(
                    frame,
                    f"Samples: {count}/{SAMPLES_REQUIRED}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "Register Student - Press Q to cancel",
                    frame
                )

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

        if count >= SAMPLES_REQUIRED:
            self.update_student_count()
            self.update_dashboard_stats()

            messagebox.showinfo(
                "Registration Complete",
                f"Registered {name} ({roll}) successfully.\n\n"
                "Click Train Face Model before taking attendance.",
                parent=self.root
            )

        else:
            with connect_db() as conn:
                conn.execute(
                    "DELETE FROM students WHERE rowid = ?",
                    (student_id,)
                )
                conn.commit()

            shutil.rmtree(folder, ignore_errors=True)
            self.update_student_count()
            self.update_dashboard_stats()

            messagebox.showwarning(
                "Registration Cancelled",
                "Face capture was incomplete. "
                "The incomplete registration was removed.",
                parent=self.root
            )

    # DELETE STUDENT AND ALL ASSOCIATED RECORDS

    def delete_student(self):
        if self.running:
            messagebox.showwarning(
                "Camera Running",
                "Stop the attendance camera before deleting a student.",
                parent=self.root
            )
            return

        roll = simpledialog.askstring(
            "Delete Student",
            "Enter the roll number of the student to delete:",
            parent=self.root
        )

        if not roll or not roll.strip():
            return

        roll = roll.strip()

        try:
            with connect_db() as conn:
                student = conn.execute("""
                    SELECT rowid, roll, name
                    FROM students
                    WHERE TRIM(roll) = ?
                    LIMIT 1
                """, (roll,)).fetchone()

                if student is None:
                    messagebox.showwarning(
                        "Student Not Found",
                        f"No student found with roll number '{roll}'.",
                        parent=self.root
                    )
                    return

                student_id, student_roll, student_name = student

                confirm = messagebox.askyesno(
                    "Confirm Deletion",
                    f"Delete all details for this student?\n\n"
                    f"Name: {student_name}\n"
                    f"Roll Number: {student_roll}\n\n"
                    "This removes registration, attendance history, "
                    "saved face images, and matching rows in Excel reports.",
                    parent=self.root
                )

                if not confirm:
                    return

                conn.execute(
                    "DELETE FROM attendance WHERE TRIM(roll) = ?",
                    (student_roll,)
                )

                conn.execute(
                    "DELETE FROM students WHERE rowid = ?",
                    (student_id,)
                )

                conn.commit()

        except sqlite3.Error as exc:
            messagebox.showerror(
                "Database Error", str(exc), parent=self.root
            )
            return

        shutil.rmtree(
            os.path.join(DATASET_DIR, str(student_id)),
            ignore_errors=True
        )

        # Remove matching rows from previously exported reports.
        try:
            from openpyxl import load_workbook

            for filename in os.listdir(REPORT_DIR):
                if not filename.lower().endswith(".xlsx"):
                    continue

                path = os.path.join(REPORT_DIR, filename)

                try:
                    workbook = load_workbook(path)
                    changed = False

                    for worksheet in workbook.worksheets:
                        roll_column = None

                        for cell in worksheet[1]:
                            heading = str(
                                cell.value or ""
                            ).strip().lower()

                            if heading in (
                                "roll number", "roll", "student roll"
                            ):
                                roll_column = cell.column
                                break

                        if roll_column is None:
                            continue

                        for row_number in range(
                            worksheet.max_row, 1, -1
                        ):
                            value = worksheet.cell(
                                row=row_number,
                                column=roll_column
                            ).value

                            if (
                                value is not None
                                and str(value).strip() == student_roll
                            ):
                                worksheet.delete_rows(row_number, 1)
                                changed = True

                    if changed:
                        workbook.save(path)

                    workbook.close()

                except Exception as exc:
                    print(
                        f"Could not clean report {filename}: {exc}"
                    )

        except Exception as exc:
            print("Could not clean Excel reports:", exc)

        # Remove old model so deleted identities cannot be used.
        
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError as exc:
                messagebox.showwarning(
                    "Model File Warning",
                    f"Student deleted, but old model could not be removed:\n{exc}",
                    parent=self.root
                )

        # Retrain using the remaining registered students.
        try:
            if get_students():
                train_model()
        except Exception as exc:
            messagebox.showwarning(
                "Model Training Warning",
                f"Student deleted, but model retraining failed:\n{exc}\n\n"
                "Click Train Face Model before attendance.",
                parent=self.root
            )

        self.students = get_students()
        self.update_student_count()
        self.update_dashboard_stats()

        self.status_text.set(
            f"Deleted {student_name} ({student_roll}) and associated records."
        )

        messagebox.showinfo(
            "Student Deleted",
            f"{student_name} ({student_roll}) was deleted.\n\n"
            "You can register the student again using the same roll number.",
            parent=self.root
        )

    # TRAIN FACE MODEL

    def train_faces(self):
        if self.running:
            messagebox.showwarning(
                "Camera Running",
                "Stop the camera before training.",
                parent=self.root
            )
            return

        try:
            if train_model():
                messagebox.showinfo(
                    "Training Complete",
                    "Face model trained successfully.",
                    parent=self.root
                )
                self.status_text.set("Face model trained successfully.")

            else:
                messagebox.showwarning(
                    "No Face Samples",
                    "Register a student and capture face samples first.",
                    parent=self.root
                )

        except Exception as exc:
            messagebox.showerror(
                "Training Error", str(exc), parent=self.root
            )

    # START ATTENDANCE CAMERA

    def start_camera(self):
        if self.running:
            return

        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning(
                "Model Missing",
                "Register students and train the face model first.",
                parent=self.root
            )
            return

        try:
            self.recognizer = create_recognizer()
            self.recognizer.read(MODEL_PATH)

        except Exception as exc:
            messagebox.showerror(
                "Model Error", str(exc), parent=self.root
            )
            return

        self.students = get_students()

        if not self.students:
            messagebox.showwarning(
                "No Students",
                "Register students before taking attendance.",
                parent=self.root
            )
            return

        # Built-in laptop camera.
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Could not open the laptop camera. Close other camera "
                "applications and try again.",
                parent=self.root
            )
            return

        self.running = True
        self.last_recognized_id = None
        self.recognition_count = 0

        self.status_text.set(
            "Camera running. Look at the camera to mark attendance."
        )

        self.update_camera()

    # CAMERA RECOGNITION

    def update_camera(self):
        if not self.running or self.camera is None:
            return

        ok, frame = self.camera.read()

        if not ok:
            self.stop_camera()
            self.status_text.set("Unable to read from camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        recognized_student = None

        for (x, y, w, h) in faces:
            face = cv2.resize(
                gray[y:y + h, x:x + w],
                FACE_SIZE
            )

            label, distance = self.recognizer.predict(face)
            student = self.students.get(label)

            if student and distance < THRESHOLD:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 200, 0), 2
                )

                cv2.putText(
                    frame,
                    f"{student['name']} ({student['roll']})",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0),
                    2
                )

                if self.last_recognized_id == label:
                    self.recognition_count += 1
                else:
                    self.last_recognized_id = label
                    self.recognition_count = 1

                if self.recognition_count >= 3:
                    recognized_student = student
                    break

            else:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 0, 255), 2
                )

                cv2.putText(
                    frame,
                    "Unknown Face",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                self.last_recognized_id = None
                self.recognition_count = 0

        if recognized_student is not None:
            result_type, result = mark_attendance(recognized_student)

            self.stop_camera()

            if result_type == "success":
                play_camera_sound()

                now = datetime.now()

                self.update_dashboard_stats()

                self.status_text.set(
                    f"Attendance saved for {recognized_student['name']} "
                    f"({recognized_student['roll']}) at {result}."
                )

                messagebox.showinfo(
                    "Attendance Recorded",
                    f"Student: {recognized_student['name']}\n"
                    f"Roll Number: {recognized_student['roll']}\n"
                    f"Date: {now.strftime('%Y-%m-%d')}\n"
                    f"Time: {result}\n\n"
                    "Attendance saved successfully.\n"
                    "Camera stopped. Start it again for the next student.",
                    parent=self.root
                )

            else:
                self.status_text.set(f"Attendance error: {result}")

                messagebox.showerror(
                    "Attendance Error", str(result), parent=self.root
                )

            return

        # Display live video in the dashboard.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((850, 430))

        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        if self.running:
            self.root.after(20, self.update_camera)

    # STOP CAMERA

    def stop_camera(self):
        self.running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        if hasattr(self, "video_label"):
            self.video_label.configure(
                image="",
                text="Camera stopped",
                bg=SIDEBAR,
                fg="#C7CBE0"
            )
            self.video_label.image = None

    # GENERIC TABLE VIEW

    def show_table(self, title, columns, rows):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("900x480")
        window.configure(bg=BG)

        tk.Label(
            window,
            text=title,
            font=("Segoe UI", 17, "bold"),
            bg=BG,
            fg=TEXT
        ).pack(anchor="w", padx=16, pady=(14, 4))

        tk.Label(
            window,
            text=f"Total records: {len(rows)}",
            font=("Segoe UI", 10),
            bg=BG,
            fg=MUTED
        ).pack(anchor="w", padx=16, pady=(0, 6))

        frame = tk.Frame(window, bg=BG)
        frame.pack(fill="both", expand=True, padx=14, pady=10)

        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            style="Dashboard.Treeview"
        )

        vertical = ttk.Scrollbar(
            frame, orient="vertical", command=tree.yview
        )
        horizontal = ttk.Scrollbar(
            frame, orient="horizontal", command=tree.xview
        )

        tree.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set
        )

        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=190, anchor="center")

        for row in rows:
            tree.insert("", "end", values=row)
            
    # VIEW REGISTERED STUDENTS

    def view_students(self):
        try:
            with connect_db() as conn:
                rows = conn.execute("""
                    SELECT roll, name
                    FROM students
                    ORDER BY name COLLATE NOCASE
                """).fetchall()

            self.show_table(
                "Registered Students",
                ("Roll Number", "Student Name"),
                rows
            )

        except sqlite3.Error as exc:
            messagebox.showerror(
                "Database Error", str(exc), parent=self.root
            )

    # VIEW LATEST ATTENDANCE PER STUDENT PER DAY

    def view_attendance(self):
        try:
            with connect_db() as conn:
                rows = conn.execute("""
                    SELECT
                        a.roll,
                        a.name,
                        a.attendance_date,
                        a.attendance_time
                    FROM attendance AS a
                    WHERE a.id = (
                        SELECT a2.id
                        FROM attendance AS a2
                        WHERE a2.roll = a.roll
                          AND a2.attendance_date = a.attendance_date
                        ORDER BY a2.attendance_time DESC, a2.id DESC
                        LIMIT 1
                    )
                    ORDER BY
                        a.attendance_date DESC,
                        a.name COLLATE NOCASE
                """).fetchall()

            self.show_table(
                "Latest Attendance Records",
                (
                    "Roll Number",
                    "Student Name",
                    "Attendance Date",
                    "Latest Attendance Time"
                ),
                rows
            )

        except sqlite3.Error as exc:
            messagebox.showerror(
                "Database Error", str(exc), parent=self.root
            )

    # EXPORT TODAY'S ATTENDANCE TO EXCEL

    def export_excel(self):
        today = date.today().strftime("%Y-%m-%d")

        try:
            with connect_db() as conn:
                students = pd.read_sql_query("""
                    SELECT roll, name
                    FROM students
                    ORDER BY name COLLATE NOCASE
                """, conn)

                attendance = pd.read_sql_query("""
                    SELECT
                        a.roll,
                        a.attendance_date,
                        a.attendance_time
                    FROM attendance AS a
                    WHERE a.id = (
                        SELECT a2.id
                        FROM attendance AS a2
                        WHERE a2.roll = a.roll
                          AND a2.attendance_date = a.attendance_date
                        ORDER BY a2.attendance_time DESC, a2.id DESC
                        LIMIT 1
                    )
                    AND a.attendance_date = ?
                """, conn, params=(today,))

            if students.empty:
                messagebox.showwarning(
                    "No Students",
                    "Register students before exporting.",
                    parent=self.root
                )
                return

            report = students.merge(
                attendance, on="roll", how="left"
            )

            report["Status"] = report["attendance_date"].apply(
                lambda value: (
                    "Present" if pd.notna(value) else "Absent"
                )
            )

            report.rename(
                columns={
                    "roll": "Roll Number",
                    "name": "Student Name",
                    "attendance_date": "Attendance Date",
                    "attendance_time": "Latest Attendance Time"
                },
                inplace=True
            )

            report = report[
                [
                    "Roll Number",
                    "Student Name",
                    "Attendance Date",
                    "Latest Attendance Time",
                    "Status"
                ]
            ]

            filename = os.path.join(
                REPORT_DIR,
                f"attendance_{today}_"
                f"{datetime.now().strftime('%H%M%S')}.xlsx"
            )

            with pd.ExcelWriter(
                filename, engine="openpyxl"
            ) as writer:
                report.to_excel(
                    writer,
                    index=False,
                    sheet_name="Attendance"
                )

                worksheet = writer.sheets["Attendance"]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions

                for cells in worksheet.columns:
                    max_length = max(
                        len(str(cell.value or ""))
                        for cell in cells
                    )

                    worksheet.column_dimensions[
                        cells[0].column_letter
                    ].width = min(max_length + 3, 32)

            messagebox.showinfo(
                "Export Successful",
                f"Excel file created:\n{filename}\n\n"
                "The report contains today's attendance and the latest "
                "attendance time for each student.",
                parent=self.root
            )

            try:
                os.startfile(filename)
            except OSError:
                pass

        except Exception as exc:
            messagebox.showerror(
                "Excel Export Error", str(exc), parent=self.root
            )

    # OPEN REPORTS FOLDER

    def open_reports_folder(self):
        try:
            os.startfile(REPORT_DIR)
        except OSError as exc:
            messagebox.showerror(
                "Folder Error", str(exc), parent=self.root
            )

    # CLOSE APPLICATION

    def close_app(self):
        self.stop_camera()
        self.root.destroy()

# RUN APPLICATION

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    try:
        setup_database()

        app = AttendanceApp(root)

        root.deiconify()
        root.mainloop()

    except Exception as exc:
        messagebox.showerror(
            "Application Error",
            str(exc),
            parent=root
        )

        try:
            root.destroy()
        except tk.TclError:
            pass
