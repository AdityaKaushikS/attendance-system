# Face Recognition Attendance System

A Tkinter desktop application that registers students from webcam images, trains an LBPH face recognizer, and saves recognized attendance to daily CSV files.

## macOS setup

Python 3.11 or 3.12 is recommended. From Terminal, in this project folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The first camera operation should trigger a macOS permission prompt. If access was previously denied, select **Camera settings** in the app and enable Terminal (or the application used to launch Python) under **System Settings > Privacy & Security > Camera**. Restart Attendance Studio after changing permission.

## Usage

1. Enter a student ID and name.
2. Select **Take Images** and look toward the camera. Press `q` to stop early.
3. Select **Train recognition profile** after capture completes.
4. Select **Take Attendance**. Press `q` when finished.

OpenCV supplies the Haar cascade face-detector file from its installed package, so a separate XML download is not required.

## Data folders

- `TrainingImage/`: captured face samples
- `TrainingImageLabel/`: password and trained recognition model
- `StudentDetails/`: registered student CSV data
- `Attendance/`: daily attendance CSV reports

These runtime files are kept out of Git by default because they can contain biometric and personal information.
