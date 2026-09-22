"""macOS-friendly face-recognition attendance application."""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
IMAGES, LABELS = ROOT / "TrainingImage", ROOT / "TrainingImageLabel"
DETAILS, ATTENDANCE = ROOT / "StudentDetails", ROOT / "Attendance"
STUDENTS, MODEL = DETAILS / "StudentDetails.csv", LABELS / "Trainner.yml"
TARGET = 50
C = {"bg":"#F4F7FB", "card":"#FFFFFF", "nav":"#14213D", "blue":"#2563EB",
     "text":"#172033", "muted":"#667085", "line":"#DDE3EC", "green":"#16A34A", "red":"#DC2626"}


def prepare():
    for folder in (IMAGES, LABELS, DETAILS, ATTENDANCE):
        folder.mkdir(parents=True, exist_ok=True)


def students():
    result = []
    if not STUDENTS.exists():
        return result
    with STUDENTS.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row or not row[0].strip().isdigit():
                continue
            if len(row) >= 5:  # legacy format
                serial, sid, name = row[0], row[2], row[4]
            elif len(row) >= 3:
                serial, sid, name = row[:3]
            else:
                continue
            result.append({"serial": int(serial), "id": sid.strip(), "name": name.strip()})
    return result


def add_student(serial, sid, name):
    header = not STUDENTS.exists() or STUDENTS.stat().st_size == 0
    with STUDENTS.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(["Serial", "ID", "Name"])
        writer.writerow([serial, sid, name])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        prepare()
        self.title("Attendance Studio")
        self.geometry("1120x720")
        self.minsize(960, 640)
        self.configure(bg=C["bg"])
        self.protocol("WM_DELETE_WINDOW", self.close)
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade))
        self.camera = self.job = self.mode = self.photo = self.pending = self.recognizer = None
        self.samples = self.frames = 0
        self.marked = set()
        self._styles()
        self._shell()
        self.registration_page()
        self._clock()

    def _styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TEntry", padding=10, fieldbackground="white")
        s.configure("Primary.TButton", padding=(17,11), background=C["blue"], foreground="white", borderwidth=0, font=("Helvetica Neue",12,"bold"))
        s.map("Primary.TButton", background=[("active", "#1D4ED8"), ("disabled", "#9DB7EE")])
        s.configure("Secondary.TButton", padding=(14,10), background="#E8EEF8", foreground=C["text"], borderwidth=0, font=("Helvetica Neue",11,"bold"))
        s.configure("Treeview", rowheight=34, borderwidth=0, font=("Helvetica Neue",11))
        s.configure("Treeview.Heading", background="#EEF2F7", font=("Helvetica Neue",11,"bold"))
        s.configure("Horizontal.TProgressbar", background=C["blue"], troughcolor="#E7ECF3")

    def _shell(self):
        top = tk.Frame(self, bg=C["nav"], height=80); top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top,text="Attendance Studio",bg=C["nav"],fg="white",font=("Helvetica Neue",24,"bold")).pack(side="left",padx=30)
        self.time = tk.Label(top,bg=C["nav"],fg="#CBD5E1",font=("Helvetica Neue",12)); self.time.pack(side="right",padx=30)
        body=tk.Frame(self,bg=C["bg"]); body.pack(fill="both",expand=True)
        nav=tk.Frame(body,bg="white",width=220,highlightthickness=1,highlightbackground=C["line"]); nav.pack(side="left",fill="y"); nav.pack_propagate(False)
        tk.Label(nav,text="WORKSPACE",bg="white",fg=C["muted"],font=("Helvetica Neue",10,"bold")).pack(anchor="w",padx=22,pady=(28,10))
        self.regnav=self._nav(nav,"＋  Register student",self.registration_page)
        self.attnav=self._nav(nav,"✓  Take attendance",self.attendance_page)
        self.count=tk.Label(nav,bg="white",fg=C["muted"],justify="left",font=("Helvetica Neue",11)); self.count.pack(anchor="w",padx=22,pady=24)
        ttk.Button(nav,text="Camera settings",style="Secondary.TButton",command=self.camera_settings).pack(side="bottom",fill="x",padx=18,pady=22)
        self.content=tk.Frame(body,bg=C["bg"]); self.content.pack(fill="both",expand=True,padx=28,pady=22)

    def _nav(self,parent,text,command):
        b=tk.Button(parent,text=text,command=command,anchor="w",bg="white",fg=C["text"],activebackground="#EAF1FF",relief="flat",bd=0,padx=22,pady=13,font=("Helvetica Neue",12,"bold")); b.pack(fill="x"); return b

    def _clear(self, active):
        self.stop(False)
        for child in self.content.winfo_children(): child.destroy()
        for button in (self.regnav,self.attnav): button.configure(bg="#EAF1FF" if button is active else "white",fg=C["blue"] if button is active else C["text"])
        n=len(students()); self.count.configure(text=f"{n} registered\nstudent{'s' if n != 1 else ''}")

    def _title(self,title,subtitle):
        tk.Label(self.content,text=title,bg=C["bg"],fg=C["text"],font=("Helvetica Neue",25,"bold")).pack(anchor="w")
        tk.Label(self.content,text=subtitle,bg=C["bg"],fg=C["muted"],font=("Helvetica Neue",12)).pack(anchor="w",pady=(3,0))

    def _card(self,parent): return tk.Frame(parent,bg="white",highlightthickness=1,highlightbackground=C["line"])

    def registration_page(self):
        self._clear(self.regnav); self._title("Register a student","Capture clear face samples, then train the recognition profile.")
        area=tk.Frame(self.content,bg=C["bg"]); area.pack(fill="both",expand=True,pady=(18,0))
        form=self._card(area); form.pack(side="left",fill="y",padx=(0,18)); form.configure(width=330); form.pack_propagate(False)
        for text, attr in (("Student ID","idbox"),("Full name","namebox")):
            tk.Label(form,text=text,bg="white",fg=C["text"],font=("Helvetica Neue",11,"bold")).pack(anchor="w",padx=24,pady=(26 if attr=="idbox" else 18,7))
            entry=ttk.Entry(form,font=("Helvetica Neue",13)); entry.pack(fill="x",padx=24); setattr(self,attr,entry)
        tk.Label(form,text="Face the camera in even lighting. Slowly turn your head while samples are captured.",wraplength=275,justify="left",bg="white",fg=C["muted"],font=("Helvetica Neue",10)).pack(anchor="w",padx=24,pady=20)
        self.capture=ttk.Button(form,text="Start face capture",style="Primary.TButton",command=self.start_capture); self.capture.pack(fill="x",padx=24,pady=(0,10))
        ttk.Button(form,text="Train recognition profile",style="Secondary.TButton",command=self.train).pack(fill="x",padx=24)
        ttk.Button(form,text="Stop camera",style="Secondary.TButton",command=self.stop).pack(fill="x",padx=24,pady=10)
        card=self._card(area); card.pack(side="left",fill="both",expand=True)
        self.preview=tk.Label(card,text="Camera preview\n\nSelect “Start face capture” to begin",bg="#101828",fg="#94A3B8",font=("Helvetica Neue",14)); self.preview.pack(fill="both",expand=True,padx=15,pady=(15,8))
        self.progress=ttk.Progressbar(card,maximum=TARGET); self.progress.pack(fill="x",padx=18,pady=5)
        self.status=tk.Label(card,text="Ready",bg="white",fg=C["muted"],anchor="w",font=("Helvetica Neue",11)); self.status.pack(fill="x",padx=18,pady=(0,14))

    def attendance_page(self):
        self._clear(self.attnav); self._title("Take attendance","Each recognized student is recorded once per session.")
        bar=tk.Frame(self.content,bg=C["bg"]); bar.pack(fill="x",pady=(15,12))
        ttk.Button(bar,text="Start camera",style="Primary.TButton",command=self.start_attendance).pack(side="left")
        ttk.Button(bar,text="Stop camera",style="Secondary.TButton",command=self.stop).pack(side="left",padx=10)
        self.status=tk.Label(bar,text="Ready",bg=C["bg"],fg=C["muted"],font=("Helvetica Neue",11)); self.status.pack(side="right")
        pane=tk.PanedWindow(self.content,orient="horizontal",sashwidth=8,bg=C["bg"],bd=0); pane.pack(fill="both",expand=True)
        camera=self._card(pane); records=self._card(pane); pane.add(camera,minsize=360); pane.add(records,minsize=360)
        self.preview=tk.Label(camera,text="Camera preview",bg="#101828",fg="#94A3B8",font=("Helvetica Neue",14)); self.preview.pack(fill="both",expand=True,padx=14,pady=14)
        self.table=ttk.Treeview(records,columns=("id","name","time"),show="headings")
        for col,title,width in (("id","ID",90),("name","Name",160),("time","Time",95)): self.table.heading(col,text=title); self.table.column(col,width=width,anchor="w")
        self.table.pack(fill="both",expand=True,padx=14,pady=14)

    def open_camera(self):
        self.stop(False)
        backends=[cv2.CAP_AVFOUNDATION,cv2.CAP_ANY] if sys.platform=="darwin" else [cv2.CAP_ANY]
        for backend in backends:
            camera=cv2.VideoCapture(0,backend)
            if camera.isOpened(): self.camera=camera; return True
            camera.release()
        self.status.configure(text="Camera blocked by macOS",fg=C["red"])
        if messagebox.askyesno("Camera access needed","macOS has blocked camera access. Open Camera privacy settings now?\n\nEnable Terminal (or the app used to launch Python), then restart this app."): self.camera_settings()
        return False

    def camera_settings(self):
        if sys.platform=="darwin": subprocess.run(["open","x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"],check=False)
        else: messagebox.showinfo("Camera settings","Allow camera access for the application used to launch Python.")

    def start_capture(self):
        sid=self.idbox.get().strip(); name=" ".join(self.namebox.get().split())
        if not sid: messagebox.showwarning("Student ID required","Enter a student ID."); return
        if not name or not all(x.replace("-","").replace("'","").isalpha() for x in name.split()): messagebox.showwarning("Valid name required","Use letters, spaces, apostrophes, or hyphens."); return
        if any(str(x["id"]).casefold()==sid.casefold() for x in students()): messagebox.showwarning("ID already registered","Use a unique student ID."); return
        if not self.open_camera(): return
        serial=max((x["serial"] for x in students()),default=0)+1
        self.pending=(serial,sid,name); self.samples=self.frames=0; self.progress["value"]=0; self.mode="capture"; self.capture.configure(state="disabled")
        self.status.configure(text="Camera active — keep your face inside the frame",fg=C["blue"]); self.poll()

    def start_attendance(self):
        if not MODEL.exists(): messagebox.showwarning("Profile not trained","Register a student and train the profile first."); return
        self.recognizer=cv2.face.LBPHFaceRecognizer_create(); self.recognizer.read(str(MODEL))
        if not self.open_camera(): return
        self.marked=set(); self.mode="attendance"; self.status.configure(text="Scanning…",fg=C["blue"]); self.poll()

    def poll(self):
        if self.camera is None: return
        ok,frame=self.camera.read()
        if not ok: self.status.configure(text="Camera stopped returning frames",fg=C["red"]); self.stop(False); return
        frame=cv2.flip(frame,1); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); self.frames+=1
        faces=self.detector.detectMultiScale(gray,1.2,5,minSize=(90,90))
        for x,y,w,h in faces:
            label="Face detected"; color=(54,191,109)
            if self.mode=="capture" and self.frames%3==0 and self.samples<TARGET:
                serial,sid,name=self.pending; self.samples+=1
                cv2.imwrite(str(IMAGES/f"{name}.{serial}.{sid}.{self.samples}.jpg"),gray[y:y+h,x:x+w]); self.progress["value"]=self.samples
                self.status.configure(text=f"Capturing — {self.samples} of {TARGET}",fg=C["blue"])
            elif self.mode=="attendance":
                serial,confidence=self.recognizer.predict(gray[y:y+h,x:x+w]); student=next((s for s in students() if s["serial"]==serial),None)
                if student and confidence<65:
                    label=student["name"]
                    if serial not in self.marked: self.record(student); self.marked.add(serial)
                else: label="Unknown"; color=(70,70,230)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2); cv2.putText(frame,str(label),(x,max(28,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        self.show_frame(frame)
        if self.mode=="capture" and self.samples>=TARGET:
            serial,sid,name=self.pending; add_student(serial,sid,name); self.status.configure(text="Capture complete — train the profile",fg=C["green"]); self.stop(False)
            messagebox.showinfo("Capture complete",f"Saved {TARGET} face samples for {name}.\n\nNow train the recognition profile."); return
        self.job=self.after(30,self.poll)

    def show_frame(self,frame):
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); image.thumbnail((max(320,self.preview.winfo_width()-8),max(240,self.preview.winfo_height()-8)),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(image); self.preview.configure(image=self.photo,text="")

    def train(self):
        faces=[]; ids=[]
        for path in IMAGES.glob("*.jpg"):
            parts=path.stem.split(".")
            if len(parts)>=4 and parts[-3].isdigit():
                image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if image is not None: faces.append(image); ids.append(int(parts[-3]))
        if not faces: messagebox.showwarning("No samples","Capture face samples first."); return
        model=cv2.face.LBPHFaceRecognizer_create(); model.train(faces,np.asarray(ids)); model.save(str(MODEL))
        self.status.configure(text=f"Profile trained for {len(set(ids))} student(s)",fg=C["green"]); messagebox.showinfo("Training complete","The profile is ready for attendance.")

    def record(self,student):
        now=dt.datetime.now(); path=ATTENDANCE/f"Attendance_{now:%d-%m-%Y}.csv"; header=not path.exists() or path.stat().st_size==0
        with path.open("a",newline="",encoding="utf-8") as file:
            writer=csv.writer(file)
            if header: writer.writerow(["ID","Name","Date","Time"])
            writer.writerow([student["id"],student["name"],now.strftime("%d-%m-%Y"),now.strftime("%H:%M:%S")])
        self.table.insert("","end",values=(student["id"],student["name"],now.strftime("%H:%M:%S"))); self.status.configure(text=f"Marked {student['name']}",fg=C["green"])

    def stop(self,clear=True):
        if self.job:
            try: self.after_cancel(self.job)
            except tk.TclError: pass
            self.job=None
        if self.camera is not None: self.camera.release(); self.camera=None
        self.mode=None
        if hasattr(self,"capture") and self.capture.winfo_exists(): self.capture.configure(state="normal")
        if clear and hasattr(self,"preview") and self.preview.winfo_exists(): self.preview.configure(image="",text="Camera stopped"); self.photo=None

    def _clock(self): self.time.configure(text=dt.datetime.now().strftime("%A, %d %B  •  %I:%M:%S %p")); self.after(1000,self._clock)
    def close(self): self.stop(); self.destroy()


if __name__ == "__main__":
    App().mainloop()
