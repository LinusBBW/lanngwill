import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, simpledialog, Checkbutton, BooleanVar, Listbox, Scrollbar, Entry, Button, Frame, Label
import numpy as np
import os
import threading
import time
import pickle
import psutil
import platform
from sklearn.metrics.pairwise import cosine_similarity

# PyAutoGUI für zuverlässige Steuerung
try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
    pyautogui.FAILSAFE = False
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class SpecificFaceSecurity:
    def __init__(self):
        # MediaPipe Setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=5,
            refine_landmarks=True, min_detection_confidence=0.5)

        # Face Recognition Setup
        self.known_faces = {}
        self.video_capture = cv2.VideoCapture(0)
        self.alert_cooldown = 0
        self.recognition_threshold = 0.85
        self.min_samples_per_person = 3

        # 🎯 ERWEITERTE SPEZIFISCHE SICHERHEITSEINSTELLUNGEN
        self.security_actions = {
            # Basis-Aktionen
            'minimize_all': False,
            'mute_system': True,
            'lock_screen': False,

            # Spezifische Apps (Liste von Prozessnamen)
            'specific_processes': [
                'chrome.exe',
                'firefox.exe',
                'spotify.exe',
                'discord.exe'
            ],

            # Fenster nach Titel schließen (für YouTube, Netflix, etc.)
            'window_titles': [
                'YouTube',
                'Netflix',
                'Disney+',
                'Twitch',
                'TikTok',
                'Instagram'
            ],

            # Browser-spezifische Aktionen
            'browser_actions': {
                'close_youtube_tabs': True,
                'close_social_media_tabs': True,
                'close_streaming_tabs': True,
                'close_all_browsers': False
            }
        }

        # Vordefinierte Listen für schnelle Auswahl
        self.predefined_apps = {
            '🌐 Browser': ['chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'brave.exe'],
            '🎵 Media Player': ['spotify.exe', 'vlc.exe', 'wmplayer.exe', 'musicbee.exe', 'itunes.exe'],
            '💬 Social Apps': ['discord.exe', 'whatsapp.exe', 'telegram.exe', 'skype.exe', 'zoom.exe'],
            '🎮 Gaming': ['steam.exe', 'epicgameslauncher.exe', 'battle.net.exe', 'uplay.exe'],
            '📝 Office': ['winword.exe', 'excel.exe', 'powerpoint.exe', 'notepad.exe'],
            '🛒 Shopping': ['amazon.exe', 'ebay.exe'],
        }

        self.predefined_titles = {
            '📺 Streaming': ['YouTube', 'Netflix', 'Disney+', 'Prime Video', 'Twitch', 'Crunchyroll'],
            '📱 Social Media': ['Facebook', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'Reddit'],
            '🎵 Musik': ['Spotify', 'Apple Music', 'YouTube Music', 'SoundCloud'],
            '🎮 Gaming': ['Steam', 'Epic Games', 'Battle.net', 'Origin', 'Uplay'],
            '🛒 Shopping': ['Amazon', 'eBay', 'AliExpress', 'Zalando']
        }

        self.load_known_faces()
        self.load_security_settings()

    def extract_enhanced_features(self, landmarks, frame_shape):
        """Feature-Extraktion (gleich wie vorher)"""
        if not landmarks:
            return None

        key_points = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321,
            308, 324, 318,
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            18, 175, 199, 175, 199, 175, 199
        ]

        features = []
        for point_idx in key_points:
            if point_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[point_idx]
                features.extend([landmark.x, landmark.y, landmark.z])

        if len(landmarks.landmark) > 300:
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[362]
            eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)

            nose_tip = landmarks.landmark[1]
            mouth_center = landmarks.landmark[13]
            nose_mouth_dist = np.sqrt((nose_tip.x - mouth_center.x) ** 2 + (nose_tip.y - mouth_center.y) ** 2)

            top = landmarks.landmark[10]
            bottom = landmarks.landmark[175]
            left = landmarks.landmark[234]
            right = landmarks.landmark[454]

            face_height = abs(top.y - bottom.y)
            face_width = abs(left.x - right.x)
            face_ratio = face_height / (face_width + 0.001)

            features.extend([eye_distance, nose_mouth_dist, face_ratio])

        return np.array(features)

    def compare_faces_improved(self, features1, features2):
        """Gesichtserkennung"""
        if features1 is None or features2 is None:
            return 0.0

        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]

        similarity = cosine_similarity([features1], [features2])[0][0]
        return similarity

    def recognize_face(self, current_features):
        """Erkennt Gesicht"""
        if current_features is None:
            return "EINDRINGLING", 0.0

        best_match = "EINDRINGLING"
        best_score = 0.0

        for name, face_samples in self.known_faces.items():
            similarities = []
            for sample_features in face_samples:
                sim = self.compare_faces_improved(current_features, sample_features)
                similarities.append(sim)

            if similarities:
                max_similarity = max(similarities)
                avg_similarity = np.mean(similarities)
                final_score = 0.7 * max_similarity + 0.3 * avg_similarity

                if final_score > best_score and final_score > self.recognition_threshold:
                    best_match = name
                    best_score = final_score

        return best_match, best_score

    # === SPEZIFISCHE SICHERHEITSFUNKTIONEN ===

    def close_specific_process(self, process_name):
        """Schließt einen spezifischen Prozess"""
        try:
            closed = False
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'].lower() == process_name.lower():
                    proc.terminate()
                    print(f"🚫 {process_name} geschlossen!")
                    closed = True
            return closed
        except Exception as e:
            print(f"❌ Fehler beim Schließen von {process_name}: {e}")
            return False

    def close_windows_by_title(self, title_keywords):
        """Schließt Fenster die bestimmte Titel enthalten"""
        if not PYAUTOGUI_AVAILABLE:
            print("❌ PyAutoGUI nicht verfügbar für Fenster-Management")
            return False

        try:
            if platform.system() == "Windows":
                import ctypes
                from ctypes import wintypes

                closed_windows = []

                def enum_windows_callback(hwnd, lParam):
                    if ctypes.windll.user32.IsWindowVisible(hwnd):
                        # Fenstertitel abrufen
                        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                            window_title = buff.value

                            # Prüfen ob eines der Keywords im Titel steht
                            for keyword in title_keywords:
                                if keyword.lower() in window_title.lower():
                                    # Fenster schließen
                                    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
                                    closed_windows.append(f"{keyword} ({window_title[:30]}...)")
                                    break
                    return True

                EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum_windows_callback), 0)

                if closed_windows:
                    print(f"🚫 Fenster geschlossen: {', '.join(closed_windows)}")
                    return True

        except Exception as e:
            print(f"❌ Fehler beim Schließen von Fenstern: {e}")

        return False

    def close_browser_tabs_smart(self):
        """Intelligentes Schließen von Browser-Tabs"""
        if not PYAUTOGUI_AVAILABLE:
            print("❌ PyAutoGUI nicht verfügbar für Tab-Management")
            return False

        try:
            actions_taken = []

            # YouTube Tabs schließen
            if self.security_actions['browser_actions']['close_youtube_tabs']:
                # Ctrl+Shift+A (Chrome Extensions) oder direkt Ctrl+W für aktive Tabs
                pyautogui.hotkey('ctrl', 'l')  # Adressleiste fokussieren
                time.sleep(0.2)
                pyautogui.typewrite('chrome://extensions/')  # Zu Extensions
                time.sleep(0.5)

                # Zurück und YouTube-Tabs schließen
                for i in range(10):  # Max 10 Tabs durchgehen
                    pyautogui.hotkey('ctrl', 'l')
                    time.sleep(0.1)
                    pyautogui.hotkey('ctrl', 'c')  # URL kopieren
                    time.sleep(0.1)

                    # Vereinfachter Ansatz: Einfach verdächtige Tabs schließen
                    pyautogui.hotkey('ctrl', 'w')  # Tab schließen
                    time.sleep(0.2)

                actions_taken.append("YouTube Tabs")

            # Alle Browser schließen (falls gewählt)
            if self.security_actions['browser_actions']['close_all_browsers']:
                for browser in ['chrome.exe', 'firefox.exe', 'msedge.exe']:
                    if self.close_specific_process(browser):
                        actions_taken.append(f"Browser ({browser})")

            if actions_taken:
                print(f"🌐 Browser-Aktionen: {', '.join(actions_taken)}")
                return True

        except Exception as e:
            print(f"❌ Fehler beim Browser-Management: {e}")

        return False

    def minimize_all_windows_simple(self):
        """Minimiert alle Fenster"""
        if not PYAUTOGUI_AVAILABLE:
            return False

        try:
            pyautogui.hotkey('win', 'd')
            print("📦 Desktop angezeigt!")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Minimieren: {e}")
            return False

    def mute_system_simple(self):
        """Schaltet System stumm"""
        if not PYAUTOGUI_AVAILABLE:
            return False

        try:
            pyautogui.press('volumemute')
            print("🔇 System stumm geschaltet!")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Stummschalten: {e}")
            return False

    def execute_security_measures(self):
        """Führt alle spezifischen Sicherheitsmaßnahmen aus"""
        print("\n" + "=" * 60)
        print("🎯 SPEZIFISCHE SICHERHEITSMASSNAHMEN AKTIVIERT! 🎯")
        print("=" * 60)

        actions_taken = []

        # 1. Spezifische Prozesse schließen
        if self.security_actions['specific_processes']:
            print("🚫 Schließe spezifische Apps...")
            closed_processes = []
            for process in self.security_actions['specific_processes']:
                if self.close_specific_process(process):
                    closed_processes.append(process.replace('.exe', ''))
            if closed_processes:
                actions_taken.append(f"Apps geschlossen: {', '.join(closed_processes)}")

        # 2. Fenster nach Titel schließen
        if self.security_actions['window_titles']:
            print("🪟 Schließe Fenster nach Titel...")
            if self.close_windows_by_title(self.security_actions['window_titles']):
                actions_taken.append(f"Fenster geschlossen: {', '.join(self.security_actions['window_titles'])}")

        # 3. Browser-spezifische Aktionen
        print("🌐 Browser-Management...")
        if self.close_browser_tabs_smart():
            actions_taken.append("Browser-Tabs verwaltet")

        # 4. System-Aktionen
        if self.security_actions['mute_system']:
            if self.mute_system_simple():
                actions_taken.append("System stumm geschaltet")

        if self.security_actions['minimize_all']:
            if self.minimize_all_windows_simple():
                actions_taken.append("Alle Fenster minimiert")

        # Ergebnis anzeigen
        if actions_taken:
            print("✅ ERFOLGREICH AUSGEFÜHRT:")
            for action in actions_taken:
                print(f"   • {action}")
        else:
            print("❌ Keine Sicherheitsmaßnahmen erfolgreich")

        print("=" * 60 + "\n")

    def configure_security_settings(self):
        """Erweiterte GUI für spezifische Sicherheitseinstellungen"""
        settings_window = tk.Tk()
        settings_window.title("🎯 Spezifische Sicherheitseinstellungen")
        settings_window.geometry("800x700")
        settings_window.configure(bg='white')

        # Titel
        title_label = tk.Label(settings_window, text="🎯 Spezifische Sicherheitseinstellungen",
                               font=("Arial", 16, "bold"), bg='white', fg='darkblue')
        title_label.pack(pady=10)

        # Notebook-ähnliche Tabs mit Frames
        notebook_frame = tk.Frame(settings_window, bg='white')
        notebook_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Tab-Buttons
        tab_frame = tk.Frame(notebook_frame, bg='lightgray')
        tab_frame.pack(fill='x', pady=(0, 10))

        current_tab = tk.StringVar(value="processes")

        def show_tab(tab_name):
            current_tab.set(tab_name)
            for widget in content_frame.winfo_children():
                widget.destroy()

            if tab_name == "processes":
                create_processes_tab()
            elif tab_name == "windows":
                create_windows_tab()
            elif tab_name == "browser":
                create_browser_tab()
            elif tab_name == "system":
                create_system_tab()

        # Tab-Buttons erstellen
        tk.Button(tab_frame, text="📱 Apps", command=lambda: show_tab("processes"),
                  bg='lightblue', font=("Arial", 10, "bold")).pack(side='left', padx=2)
        tk.Button(tab_frame, text="🪟 Fenster", command=lambda: show_tab("windows"),
                  bg='lightgreen', font=("Arial", 10, "bold")).pack(side='left', padx=2)
        tk.Button(tab_frame, text="🌐 Browser", command=lambda: show_tab("browser"),
                  bg='lightyellow', font=("Arial", 10, "bold")).pack(side='left', padx=2)
        tk.Button(tab_frame, text="⚙️ System", command=lambda: show_tab("system"),
                  bg='lightcoral', font=("Arial", 10, "bold")).pack(side='left', padx=2)

        # Content Frame
        content_frame = tk.Frame(notebook_frame, bg='white', relief='sunken', bd=2)
        content_frame.pack(fill='both', expand=True)

        def create_processes_tab():
            tk.Label(content_frame, text="📱 Spezifische Apps schließen",
                     font=("Arial", 14, "bold"), bg='white').pack(pady=10)

            # Vordefinierte Apps
            predefined_frame = tk.Frame(content_frame, bg='white')
            predefined_frame.pack(fill='x', padx=20, pady=10)

            tk.Label(predefined_frame, text="Schnellauswahl:", font=("Arial", 10, "bold"), bg='white').pack(anchor='w')

            for category, apps in self.predefined_apps.items():
                frame = tk.Frame(predefined_frame, bg='white')
                frame.pack(fill='x', pady=2)

                def add_category_apps(app_list=apps):
                    for app in app_list:
                        if app not in self.security_actions['specific_processes']:
                            self.security_actions['specific_processes'].append(app)
                    update_process_list()

                tk.Button(frame, text=f"+ {category}", command=add_category_apps,
                          bg='lightblue', font=("Arial", 8)).pack(side='left')

            # Aktuelle Liste
            list_frame = tk.Frame(content_frame, bg='white')
            list_frame.pack(fill='both', expand=True, padx=20, pady=10)

            tk.Label(list_frame, text="Aktuelle Apps (werden bei Eindringling geschlossen):",
                     font=("Arial", 10, "bold"), bg='white').pack(anchor='w')

            listbox_frame = tk.Frame(list_frame, bg='white')
            listbox_frame.pack(fill='both', expand=True)

            process_listbox = Listbox(listbox_frame, height=8)
            process_listbox.pack(side='left', fill='both', expand=True)

            scrollbar1 = Scrollbar(listbox_frame, orient='vertical')
            scrollbar1.pack(side='right', fill='y')
            process_listbox.config(yscrollcommand=scrollbar1.set)
            scrollbar1.config(command=process_listbox.yview)

            def update_process_list():
                process_listbox.delete(0, tk.END)
                for process in self.security_actions['specific_processes']:
                    process_listbox.insert(tk.END, process)

            def remove_selected_process():
                selection = process_listbox.curselection()
                if selection:
                    process = process_listbox.get(selection[0])
                    self.security_actions['specific_processes'].remove(process)
                    update_process_list()

            def add_custom_process():
                process = simpledialog.askstring("App hinzufügen",
                                                 "Prozessname eingeben (z.B. notepad.exe):")
                if process and process not in self.security_actions['specific_processes']:
                    self.security_actions['specific_processes'].append(process)
                    update_process_list()

            button_frame = tk.Frame(list_frame, bg='white')
            button_frame.pack(fill='x', pady=5)

            tk.Button(button_frame, text="➕ Hinzufügen", command=add_custom_process,
                      bg='green', fg='white', font=("Arial", 9)).pack(side='left', padx=5)
            tk.Button(button_frame, text="➖ Entfernen", command=remove_selected_process,
                      bg='red', fg='white', font=("Arial", 9)).pack(side='left', padx=5)

            update_process_list()

        def create_windows_tab():
            tk.Label(content_frame, text="🪟 Fenster nach Titel schließen",
                     font=("Arial", 14, "bold"), bg='white').pack(pady=10)

            tk.Label(content_frame, text="Fenster mit diesen Begriffen im Titel werden geschlossen:",
                     font=("Arial", 10), bg='white').pack()

            # Vordefinierte Titel
            predefined_frame = tk.Frame(content_frame, bg='white')
            predefined_frame.pack(fill='x', padx=20, pady=10)

            tk.Label(predefined_frame, text="Schnellauswahl:", font=("Arial", 10, "bold"), bg='white').pack(anchor='w')

            for category, titles in self.predefined_titles.items():
                frame = tk.Frame(predefined_frame, bg='white')
                frame.pack(fill='x', pady=2)

                def add_category_titles(title_list=titles):
                    for title in title_list:
                        if title not in self.security_actions['window_titles']:
                            self.security_actions['window_titles'].append(title)
                    update_title_list()

                tk.Button(frame, text=f"+ {category}", command=add_category_titles,
                          bg='lightgreen', font=("Arial", 8)).pack(side='left')

            # Aktuelle Liste
            list_frame = tk.Frame(content_frame, bg='white')
            list_frame.pack(fill='both', expand=True, padx=20, pady=10)

            listbox_frame = tk.Frame(list_frame, bg='white')
            listbox_frame.pack(fill='both', expand=True)

            title_listbox = Listbox(listbox_frame, height=8)
            title_listbox.pack(side='left', fill='both', expand=True)

            scrollbar2 = Scrollbar(listbox_frame, orient='vertical')
            scrollbar2.pack(side='right', fill='y')
            title_listbox.config(yscrollcommand=scrollbar2.set)
            scrollbar2.config(command=title_listbox.yview)

            def update_title_list():
                title_listbox.delete(0, tk.END)
                for title in self.security_actions['window_titles']:
                    title_listbox.insert(tk.END, title)

            def remove_selected_title():
                selection = title_listbox.curselection()
                if selection:
                    title = title_listbox.get(selection[0])
                    self.security_actions['window_titles'].remove(title)
                    update_title_list()

            def add_custom_title():
                title = simpledialog.askstring("Fenstertitel hinzufügen",
                                               "Titel oder Teilwort eingeben (z.B. YouTube):")
                if title and title not in self.security_actions['window_titles']:
                    self.security_actions['window_titles'].append(title)
                    update_title_list()

            button_frame = tk.Frame(list_frame, bg='white')
            button_frame.pack(fill='x', pady=5)

            tk.Button(button_frame, text="➕ Hinzufügen", command=add_custom_title,
                      bg='green', fg='white', font=("Arial", 9)).pack(side='left', padx=5)
            tk.Button(button_frame, text="➖ Entfernen", command=remove_selected_title,
                      bg='red', fg='white', font=("Arial", 9)).pack(side='left', padx=5)

            update_title_list()

        def create_browser_tab():
            tk.Label(content_frame, text="🌐 Browser-Management",
                     font=("Arial", 14, "bold"), bg='white').pack(pady=10)

            browser_vars = {}
            for action, enabled in self.security_actions['browser_actions'].items():
                browser_vars[action] = BooleanVar(value=enabled)

                action_names = {
                    'close_youtube_tabs': '📺 YouTube Tabs schließen',
                    'close_social_media_tabs': '📱 Social Media Tabs schließen',
                    'close_streaming_tabs': '🎬 Streaming Tabs schließen',
                    'close_all_browsers': '🚫 Alle Browser komplett schließen'
                }

                text = action_names.get(action, action.replace('_', ' ').title())
                cb = Checkbutton(content_frame, text=text, variable=browser_vars[action],
                                 font=("Arial", 10), bg='white')
                cb.pack(anchor='w', padx=20, pady=5)

            def save_browser_settings():
                for action, var in browser_vars.items():
                    self.security_actions['browser_actions'][action] = var.get()

            tk.Button(content_frame, text="💾 Browser-Einstellungen speichern",
                      command=save_browser_settings, bg='blue', fg='white',
                      font=("Arial", 10, "bold")).pack(pady=20)

        def create_system_tab():
            tk.Label(content_frame, text="⚙️ System-Aktionen",
                     font=("Arial", 14, "bold"), bg='white').pack(pady=10)

            system_vars = {}
            system_actions = ['minimize_all', 'mute_system', 'lock_screen']
            system_names = {
                'minimize_all': '📦 Alle Fenster minimieren',
                'mute_system': '🔇 System stumm schalten',
                'lock_screen': '🔒 Desktop sperren'
            }

            for action in system_actions:
                system_vars[action] = BooleanVar(value=self.security_actions.get(action, False))
                text = system_names.get(action, action.replace('_', ' ').title())
                cb = Checkbutton(content_frame, text=text, variable=system_vars[action],
                                 font=("Arial", 12), bg='white')
                cb.pack(anchor='w', padx=20, pady=10)

            def save_system_settings():
                for action, var in system_vars.items():
                    self.security_actions[action] = var.get()

            tk.Button(content_frame, text="💾 System-Einstellungen speichern",
                      command=save_system_settings, bg='purple', fg='white',
                      font=("Arial", 10, "bold")).pack(pady=20)

        # Haupt-Buttons
        main_button_frame = tk.Frame(settings_window, bg='white')
        main_button_frame.pack(fill='x', padx=20, pady=10)

        def save_all_settings():
            self.save_security_settings()
            messagebox.showinfo("✅ Gespeichert", "Alle Einstellungen wurden gespeichert!")
            settings_window.destroy()

        def test_all_security():
            messagebox.showinfo("🧪 Test", "Teste alle Sicherheitsmaßnahmen...")
            self.execute_security_measures()

        tk.Button(main_button_frame, text="💾 ALLE SPEICHERN", command=save_all_settings,
                  bg='green', fg='white', font=("Arial", 12, "bold"), padx=20).pack(side='left', padx=10)
        tk.Button(main_button_frame, text="🧪 ALLES TESTEN", command=test_all_security,
                  bg='orange', fg='white', font=("Arial", 12, "bold"), padx=20).pack(side='left', padx=10)
        tk.Button(main_button_frame, text="❌ ABBRECHEN", command=settings_window.destroy,
                  bg='red', fg='white', font=("Arial", 12, "bold"), padx=20).pack(side='left', padx=10)

        # Standard-Tab anzeigen
        show_tab("processes")
        settings_window.mainloop()

    def load_security_settings(self):
        """Lädt Sicherheitseinstellungen"""
        try:
            if os.path.exists("specific_security_settings.pkl"):
                with open("specific_security_settings.pkl", "rb") as f:
                    loaded_settings = pickle.load(f)
                    self.security_actions.update(loaded_settings)
                print("✅ Spezifische Sicherheitseinstellungen geladen")
        except Exception as e:
            print(f"❌ Fehler beim Laden der Sicherheitseinstellungen: {e}")

    def save_security_settings(self):
        """Speichert Sicherheitseinstellungen"""
        try:
            with open("specific_security_settings.pkl", "wb") as f:
                pickle.dump(self.security_actions, f)
            print("💾 Spezifische Sicherheitseinstellungen gespeichert")
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Sicherheitseinstellungen: {e}")

    def show_intruder_popup(self):
        """Pop-up für Eindringlinge"""
        messages = [
            "🎯 EINDRINGLING ERKANNT! 🎯\nSpezifische Sicherheitsmaßnahmen aktiviert!",
            "⚠️ FREMDES GESICHT! ⚠️\nMaßgeschneiderte Sicherheit läuft...",
            "🕵️ UNBEKANNTE PERSON! 🕵️\nZielgerichteter Schutz aktiv!"
        ]

        import random
        message = random.choice(messages)

        # Sicherheitsmaßnahmen in separatem Thread
        security_thread = threading.Thread(target=self.execute_security_measures)
        security_thread.daemon = True
        security_thread.start()

        # Pop-up anzeigen
        messagebox.showwarning("🎯 SPEZIFISCHER SICHERHEITSALARM 🎯", message)

    def load_known_faces(self):
        """Lädt bekannte Gesichter"""
        if os.path.exists("improved_faces.pkl"):
            try:
                with open("improved_faces.pkl", "rb") as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Geladene Personen: {list(self.known_faces.keys())}")
            except Exception as e:
                print(f"❌ Fehler beim Laden: {e}")

    def save_known_faces(self):
        """Speichert bekannte Gesichter"""
        with open("improved_faces.pkl", "wb") as f:
            pickle.dump(self.known_faces, f)
        print("💾 Gesichter gespeichert!")

    def capture_multiple_samples(self):
        """Trainingsbilder aufnehmen"""
        name = None
        root = tk.Tk()
        root.withdraw()
        name = simpledialog.askstring("Name eingeben", "Wie heißt du?")
        root.destroy()

        if not name:
            return

        print(f"\n🎯 Erfasse Trainingsbilder für {name}")
        print("📸 Bewege dein Gesicht leicht zwischen den Aufnahmen!")
        print("⏸️  Drücke SPACE für jede Aufnahme")

        if name not in self.known_faces:
            self.known_faces[name] = []

        samples_taken = 0
        samples_needed = 5

        while samples_taken < samples_needed:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                status_text = f"Sample {samples_taken + 1}/{samples_needed} - SPACE drücken"
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Kein Gesicht erkannt!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Trainingsbilder aufnehmen', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if results.multi_face_landmarks:
                    features = self.extract_enhanced_features(
                        results.multi_face_landmarks[0], frame.shape)
                    if features is not None:
                        self.known_faces[name].append(features)
                        samples_taken += 1
                        print(f"✅ Sample {samples_taken} für {name} gespeichert!")
                        if samples_taken >= samples_needed:
                            break
                        time.sleep(0.5)
            elif key == 27:
                break

        cv2.destroyWindow('Trainingsbilder aufnehmen')

        if samples_taken > 0:
            self.save_known_faces()
            print(f"🎉 {samples_taken} Trainingsbilder für {name} gespeichert!")

    def run(self):
        """Hauptschleife"""
        print("🎯 Spezifische Security Face Recognition gestartet!")
        print("=" * 70)
        print("⌨️  Steuerung:")
        print("   'q' = Beenden")
        print("   'c' = Neues Gesicht trainieren")
        print("   's' = Einstellungen anzeigen")
        print("   'p' = Spezifische Sicherheitseinstellungen öffnen")
        print("   't' = Alle Sicherheitsmaßnahmen testen")
        if PYAUTOGUI_AVAILABLE:
            print("   ✅ PyAutoGUI verfügbar - Erweiterte Funktionen aktiv!")
        else:
            print("   ⚠️  PyAutoGUI nicht verfügbar - installiere mit: pip install pyautogui")
        print("=" * 70)

        if not self.known_faces:
            response = input("🤔 Möchtest du dein Gesicht jetzt trainieren? (j/n): ")
            if response.lower() == 'j':
                self.capture_multiple_samples()

        print("\n📹 Kamera gestartet... Spezifische Sicherheit aktiv!")

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    current_features = self.extract_enhanced_features(face_landmarks, frame.shape)
                    name, confidence = self.recognize_face(current_features)

                    # SPEZIFISCHES SICHERHEITSPROTOKOLL
                    if name == "EINDRINGLING" and time.time() > self.alert_cooldown:
                        self.show_intruder_popup()
                        self.alert_cooldown = time.time() + 15  # 15 Sekunden Cooldown

                    # Gesicht zeichnen
                    color = (0, 255, 0) if name != "EINDRINGLING" else (0, 0, 255)
                    h, w, _ = frame.shape
                    x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                    y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max + 60), color, 2)
                    cv2.rectangle(frame, (x_min, y_max), (x_max, y_max + 60), color, cv2.FILLED)
                    cv2.putText(frame, name, (x_min + 5, y_max + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if name != "EINDRINGLING":
                        confidence_text = f"{confidence:.1%}"
                        cv2.putText(frame, confidence_text, (x_min + 5, y_max + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Status anzeigen
            total_people = len(self.known_faces)
            total_processes = len(self.security_actions['specific_processes'])
            total_titles = len(self.security_actions['window_titles'])

            status = f"Personen: {total_people} | Apps: {total_processes} | Fenster: {total_titles}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Spezifische Sicherheit Status
            specific_status = "SPEZIFISCHE SICHERHEIT: AKTIV"
            if PYAUTOGUI_AVAILABLE:
                specific_status += " | PyAutoGUI: OK"
            cv2.putText(frame, specific_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow('🎯 Spezifische Security Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.capture_multiple_samples()
            elif key == ord('s'):
                self.show_settings()
            elif key == ord('p'):
                self.configure_security_settings()
            elif key == ord('t'):
                print("\n🧪 Teste alle Sicherheitsmaßnahmen...")
                self.execute_security_measures()

        self.video_capture.release()
        cv2.destroyAllWindows()

    def show_settings(self):
        """Zeigt aktuelle Einstellungen"""
        print("\n" + "=" * 70)
        print("🎯 SPEZIFISCHE SICHERHEITSEINSTELLUNGEN")
        print("=" * 70)
        print(f"👥 Bekannte Personen: {len(self.known_faces)}")
        for name, samples in self.known_faces.items():
            print(f"   - {name}: {len(samples)} Trainingsbilder")

        print(f"\n📱 Spezifische Apps ({len(self.security_actions['specific_processes'])}):")
        for process in self.security_actions['specific_processes']:
            print(f"   🚫 {process}")

        print(f"\n🪟 Fenstertitel ({len(self.security_actions['window_titles'])}):")
        for title in self.security_actions['window_titles']:
            print(f"   🚫 {title}")

        print("\n🌐 Browser-Aktionen:")
        for action, enabled in self.security_actions['browser_actions'].items():
            status = "✅ AKTIV" if enabled else "❌ DEAKTIVIERT"
            action_name = action.replace('_', ' ').title()
            print(f"   {action_name}: {status}")

        print("\n⚙️ System-Aktionen:")
        system_actions = ['minimize_all', 'mute_system', 'lock_screen']
        for action in system_actions:
            enabled = self.security_actions.get(action, False)
            status = "✅ AKTIV" if enabled else "❌ DEAKTIVIERT"
            action_name = action.replace('_', ' ').title()
            print(f"   {action_name}: {status}")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import psutil
    except ImportError as e:
        print("❌ Fehlende Bibliothek!")
        if "sklearn" in str(e):
            print("📦 Installiere: pip install scikit-learn")
        if "psutil" in str(e):
            print("📦 Installiere: pip install psutil")
        exit(1)

    if not PYAUTOGUI_AVAILABLE:
        print("⚠️  Für volle Funktionalität installiere PyAutoGUI:")
        print("📦 pip install pyautogui")
        print("💡 Erweiterte Features benötigen PyAutoGUI!\n")
        time.sleep(3)

    app = SpecificFaceSecurity()
    app.run()