import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, simpledialog, Checkbutton, BooleanVar, Listbox, Scrollbar, Entry, Button, Frame, Label, \
    ttk
import numpy as np
import os
import threading
import time
import pickle
import psutil
import platform
import sys
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyAutoGUI für erweiterte Steuerung
try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.1  # Kleine Pause zwischen Aktionen
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("PyAutoGUI nicht verfügbar - Einige Funktionen werden deaktiviert")

# Windows-spezifische Imports
if platform.system() == "Windows":
    try:
        import ctypes
        from ctypes import wintypes

        WINDOWS_API_AVAILABLE = True
    except ImportError:
        WINDOWS_API_AVAILABLE = False
        logger.warning("Windows API nicht verfügbar")
else:
    WINDOWS_API_AVAILABLE = False


class ImprovedFaceSecurity:
    def __init__(self):
        logger.info("Initialisiere Improved Face Security System...")

        # MediaPipe Setup mit Fehlerbehandlung
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils

            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=5,
                refine_landmarks=True, min_detection_confidence=0.5)
            logger.info("MediaPipe erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei MediaPipe-Initialisierung: {e}")
            raise

        # Face Recognition Setup
        self.known_faces = {}
        self.video_capture = None
        self.alert_cooldown = 0
        self.recognition_threshold = 0.85
        self.min_samples_per_person = 3
        self.is_running = False

        # Erweiterte Sicherheitseinstellungen
        self.security_actions = {
            'minimize_all': False,
            'mute_system': True,
            'lock_screen': False,
            'specific_processes': [
                'chrome.exe', 'firefox.exe', 'spotify.exe', 'discord.exe'
            ],
            'window_titles': [
                'YouTube', 'Netflix', 'Disney+', 'Twitch', 'TikTok', 'Instagram'
            ],
            'browser_actions': {
                'close_youtube_tabs': True,
                'close_social_media_tabs': True,
                'close_streaming_tabs': True,
                'close_all_browsers': False
            }
        }

        # Vordefinierte Listen
        self.predefined_apps = {
            '🌐 Browser': ['chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'brave.exe'],
            '🎵 Media Player': ['spotify.exe', 'vlc.exe', 'wmplayer.exe', 'musicbee.exe', 'itunes.exe'],
            '💬 Social Apps': ['discord.exe', 'whatsapp.exe', 'telegram.exe', 'skype.exe', 'zoom.exe'],
            '🎮 Gaming': ['steam.exe', 'epicgameslauncher.exe', 'battle.net.exe', 'uplay.exe'],
            '📝 Office': ['winword.exe', 'excel.exe', 'powerpoint.exe', 'notepad.exe'],
        }

        self.predefined_titles = {
            '📺 Streaming': ['YouTube', 'Netflix', 'Disney+', 'Prime Video', 'Twitch', 'Crunchyroll'],
            '📱 Social Media': ['Facebook', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'Reddit'],
            '🎵 Musik': ['Spotify', 'Apple Music', 'YouTube Music', 'SoundCloud'],
            '🎮 Gaming': ['Steam', 'Epic Games', 'Battle.net', 'Origin', 'Uplay'],
            '🛒 Shopping': ['Amazon', 'eBay', 'AliExpress', 'Zalando']
        }

        # Initialisierung
        self.init_camera()
        self.load_known_faces()
        self.load_security_settings()

    def init_camera(self):
        """Initialisiert die Kamera mit Fehlerbehandlung"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                # Versuche andere Kamera-Indizes
                for i in range(1, 4):
                    self.video_capture = cv2.VideoCapture(i)
                    if self.video_capture.isOpened():
                        logger.info(f"Kamera {i} erfolgreich initialisiert")
                        break
                else:
                    raise Exception("Keine Kamera gefunden")
            else:
                logger.info("Standard-Kamera erfolgreich initialisiert")

            # Kamera-Einstellungen optimieren
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        except Exception as e:
            logger.error(f"Kamera-Initialisierung fehlgeschlagen: {e}")
            raise

    def extract_enhanced_features(self, landmarks, frame_shape):
        """Verbesserte Feature-Extraktion mit Fehlerbehandlung"""
        try:
            if not landmarks or not landmarks.landmark:
                return None

            # Kritische Gesichtspunkte (robuster ausgewählt)
            key_points = [
                # Augen
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                # Nase
                1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49,
                # Mund
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                # Gesichtskontur
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46
            ]

            features = []
            valid_points = 0

            for point_idx in key_points:
                if point_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[point_idx]
                    features.extend([landmark.x, landmark.y, landmark.z])
                    valid_points += 1

            # Mindestanzahl an Punkten erforderlich
            if valid_points < len(key_points) * 0.8:
                logger.warning(f"Nur {valid_points}/{len(key_points)} Gesichtspunkte erkannt")
                return None

            # Zusätzliche geometrische Features
            if len(landmarks.landmark) > 400:
                try:
                    # Augenabstand
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[362]
                    eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)

                    # Nase-Mund Abstand
                    nose_tip = landmarks.landmark[1]
                    mouth_center = landmarks.landmark[13]
                    nose_mouth_dist = np.sqrt((nose_tip.x - mouth_center.x) ** 2 + (nose_tip.y - mouth_center.y) ** 2)

                    # Gesichtsproportionen
                    top = landmarks.landmark[10]
                    bottom = landmarks.landmark[175]
                    left = landmarks.landmark[234]
                    right = landmarks.landmark[454]

                    face_height = abs(top.y - bottom.y)
                    face_width = abs(left.x - right.x)
                    face_ratio = face_height / (face_width + 0.001)

                    features.extend([eye_distance, nose_mouth_dist, face_ratio])
                except IndexError:
                    logger.warning("Fehler bei geometrischen Features - verwende Basis-Features")

            return np.array(features) if features else None

        except Exception as e:
            logger.error(f"Fehler bei Feature-Extraktion: {e}")
            return None

    def compare_faces_improved(self, features1, features2):
        """Verbesserte Gesichtsvergleichsfunktion"""
        try:
            if features1 is None or features2 is None:
                return 0.0

            # Längen angleichen
            min_len = min(len(features1), len(features2))
            if min_len == 0:
                return 0.0

            features1 = features1[:min_len]
            features2 = features2[:min_len]

            # Normalisierung
            features1 = features1 / (np.linalg.norm(features1) + 1e-8)
            features2 = features2 / (np.linalg.norm(features2) + 1e-8)

            # Cosine Similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
            return max(0.0, min(1.0, similarity))  # Clamp zwischen 0 und 1

        except Exception as e:
            logger.error(f"Fehler beim Gesichtsvergleich: {e}")
            return 0.0

    def recognize_face(self, current_features):
        """Verbesserte Gesichtserkennung"""
        try:
            if current_features is None:
                return "EINDRINGLING", 0.0

            best_match = "EINDRINGLING"
            best_score = 0.0

            for name, face_samples in self.known_faces.items():
                if not face_samples:
                    continue

                similarities = []
                for sample_features in face_samples:
                    sim = self.compare_faces_improved(current_features, sample_features)
                    similarities.append(sim)

                if similarities:
                    max_similarity = max(similarities)
                    avg_similarity = np.mean(similarities)
                    # Gewichtete Kombination: bevorzuge maximale Ähnlichkeit
                    final_score = 0.7 * max_similarity + 0.3 * avg_similarity

                    if final_score > best_score and final_score > self.recognition_threshold:
                        best_match = name
                        best_score = final_score

            return best_match, best_score

        except Exception as e:
            logger.error(f"Fehler bei Gesichtserkennung: {e}")
            return "EINDRINGLING", 0.0

    # === VERBESSERTE SICHERHEITSFUNKTIONEN ===

    def close_specific_process(self, process_name):
        """Schließt spezifische Prozesse mit verbesserter Fehlerbehandlung"""
        try:
            closed_count = 0
            processes_found = []

            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                        processes_found.append(proc.info['pid'])
                        proc.terminate()
                        closed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Warte kurz und überprüfe ob Prozesse wirklich geschlossen wurden
            if closed_count > 0:
                time.sleep(1)
                still_running = []
                for pid in processes_found:
                    try:
                        if psutil.pid_exists(pid):
                            proc = psutil.Process(pid)
                            proc.kill()  # Force kill wenn terminate nicht funktioniert
                            still_running.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                logger.info(f"Prozess {process_name}: {closed_count} Instanzen geschlossen")
                return True

            return False

        except Exception as e:
            logger.error(f"Fehler beim Schließen von {process_name}: {e}")
            return False

    def close_windows_by_title(self, title_keywords):
        """Verbesserte Fenster-Schließung mit Plattform-Support"""
        if not title_keywords:
            return False

        closed_windows = []

        try:
            if platform.system() == "Windows" and WINDOWS_API_AVAILABLE:
                closed_windows = self._close_windows_windows(title_keywords)
            elif PYAUTOGUI_AVAILABLE:
                closed_windows = self._close_windows_generic(title_keywords)
            else:
                logger.warning("Keine verfügbare Methode für Fenster-Management")
                return False

            if closed_windows:
                logger.info(f"Fenster geschlossen: {', '.join(closed_windows)}")
                return True

        except Exception as e:
            logger.error(f"Fehler beim Schließen von Fenstern: {e}")

        return False

    def _close_windows_windows(self, title_keywords):
        """Windows-spezifische Fenster-Schließung"""
        closed_windows = []

        def enum_windows_callback(hwnd, lParam):
            try:
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                        window_title = buff.value

                        for keyword in title_keywords:
                            if keyword.lower() in window_title.lower():
                                # Sanftes Schließen versuchen
                                result = ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
                                if result:
                                    closed_windows.append(f"{keyword} ({window_title[:30]})")
                                break
            except Exception as e:
                logger.error(f"Fehler beim Enum-Callback: {e}")
            return True

        try:
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum_windows_callback), 0)
        except Exception as e:
            logger.error(f"Fehler bei Windows-Enumeration: {e}")

        return closed_windows

    def _close_windows_generic(self, title_keywords):
        """Plattform-unabhängige Fenster-Schließung mit PyAutoGUI"""
        closed_windows = []

        try:
            # Alt+Tab um durch Fenster zu gehen und verdächtige zu schließen
            for i in range(10):  # Maximal 10 Fenster durchgehen
                pyautogui.hotkey('alt', 'tab')
                time.sleep(0.3)

                # Versuche Fenstertitel zu erfassen (schwierig ohne native APIs)
                # Einfachere Methode: Schließe aktives Fenster wenn es verdächtig ist
                try:
                    pyautogui.hotkey('alt', 'f4')  # Fenster schließen
                    time.sleep(0.2)
                    closed_windows.append("Unbekanntes Fenster")
                except:
                    continue

        except Exception as e:
            logger.error(f"Fehler bei generischer Fenster-Schließung: {e}")

        return closed_windows

    def close_browser_tabs_smart(self):
        """Stark verbesserte Browser-Tab-Verwaltung"""
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("PyAutoGUI nicht verfügbar für Tab-Management")
            return False

        actions_taken = []

        try:
            # Alle Browser schließen (einfachste und zuverlässigste Methode)
            if self.security_actions['browser_actions']['close_all_browsers']:
                browsers = ['chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'brave.exe']
                for browser in browsers:
                    if self.close_specific_process(browser):
                        actions_taken.append(f"Browser {browser} geschlossen")
            else:
                # Intelligente Tab-Schließung
                actions_taken.extend(self._close_browser_tabs_selective())

            if actions_taken:
                logger.info(f"Browser-Aktionen: {', '.join(actions_taken)}")
                return True

        except Exception as e:
            logger.error(f"Fehler beim Browser-Management: {e}")

        return False

    def _close_browser_tabs_selective(self):
        """Selektive Tab-Schließung in Browsern"""
        actions_taken = []

        try:
            # Versuche aktive Browser-Fenster zu finden und Tabs zu schließen
            suspicious_patterns = []

            if self.security_actions['browser_actions']['close_youtube_tabs']:
                suspicious_patterns.extend(['youtube', 'youtu.be'])
            if self.security_actions['browser_actions']['close_social_media_tabs']:
                suspicious_patterns.extend(['facebook', 'instagram', 'twitter', 'tiktok'])
            if self.security_actions['browser_actions']['close_streaming_tabs']:
                suspicious_patterns.extend(['netflix', 'disney', 'twitch', 'prime'])

            # Für jeden Browser-Typ
            for browser in ['chrome.exe', 'firefox.exe', 'msedge.exe']:
                if self._is_process_running(browser):
                    closed_tabs = self._close_tabs_in_browser(browser, suspicious_patterns)
                    if closed_tabs > 0:
                        actions_taken.append(f"{browser}: {closed_tabs} Tabs geschlossen")

        except Exception as e:
            logger.error(f"Fehler bei selektiver Tab-Schließung: {e}")

        return actions_taken

    def _is_process_running(self, process_name):
        """Überprüft ob Prozess läuft"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                    return True
        except:
            pass
        return False

    def _close_tabs_in_browser(self, browser_name, suspicious_patterns):
        """Schließt verdächtige Tabs in spezifischem Browser"""
        closed_tabs = 0

        try:
            # Fokussiere Browser-Fenster
            if platform.system() == "Windows":
                # Windows-spezifische Fokussierung
                for proc in psutil.process_iter(['pid', 'name']):
                    if proc.info['name'] and proc.info['name'].lower() == browser_name.lower():
                        # Vereinfachte Methode: Schließe mehrere Tabs
                        for i in range(5):  # Maximal 5 Tabs schließen
                            pyautogui.hotkey('ctrl', 'w')
                            time.sleep(0.3)
                            closed_tabs += 1
                        break

        except Exception as e:
            logger.error(f"Fehler beim Schließen von Tabs in {browser_name}: {e}")

        return closed_tabs

    def minimize_all_windows_simple(self):
        """Minimiert alle Fenster - plattform-unabhängig"""
        try:
            if platform.system() == "Windows" and PYAUTOGUI_AVAILABLE:
                pyautogui.hotkey('win', 'd')
                logger.info("Alle Fenster minimiert (Windows)")
                return True
            elif platform.system() == "Darwin" and PYAUTOGUI_AVAILABLE:  # macOS
                pyautogui.hotkey('f11')  # Mission Control
                logger.info("Mission Control aktiviert (macOS)")
                return True
            elif platform.system() == "Linux" and PYAUTOGUI_AVAILABLE:  # Linux
                pyautogui.hotkey('super', 'd')
                logger.info("Desktop angezeigt (Linux)")
                return True
            else:
                logger.warning("Fenster-Minimierung nicht verfügbar")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Minimieren: {e}")
            return False

    def mute_system_simple(self):
        """Schaltet System stumm - plattform-unabhängig"""
        try:
            if PYAUTOGUI_AVAILABLE:
                if platform.system() == "Windows":
                    pyautogui.press('volumemute')
                elif platform.system() == "Darwin":  # macOS
                    pyautogui.press('volumemute')
                elif platform.system() == "Linux":
                    pyautogui.press('volumemute')

                logger.info("System stumm geschaltet")
                return True
            else:
                logger.warning("Stummschaltung nicht verfügbar")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Stummschalten: {e}")
            return False

    def lock_screen_simple(self):
        """Sperrt den Bildschirm - plattform-unabhängig"""
        try:
            if PYAUTOGUI_AVAILABLE:
                if platform.system() == "Windows":
                    pyautogui.hotkey('win', 'l')
                elif platform.system() == "Darwin":  # macOS
                    pyautogui.hotkey('cmd', 'ctrl', 'q')
                elif platform.system() == "Linux":
                    pyautogui.hotkey('ctrl', 'alt', 'l')

                logger.info("Bildschirm gesperrt")
                return True
            else:
                logger.warning("Bildschirm-Sperrung nicht verfügbar")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Sperren: {e}")
            return False

    def execute_security_measures(self):
        """Führt alle Sicherheitsmaßnahmen aus mit verbesserter Fehlerbehandlung"""
        logger.info("=" * 60)
        logger.info("🎯 SICHERHEITSMASSNAHMEN WERDEN AUSGEFÜHRT")
        logger.info("=" * 60)

        actions_taken = []
        failed_actions = []

        try:
            # 1. Spezifische Prozesse schließen
            if self.security_actions['specific_processes']:
                logger.info("Schließe spezifische Apps...")
                closed_processes = []
                for process in self.security_actions['specific_processes']:
                    try:
                        if self.close_specific_process(process):
                            closed_processes.append(process.replace('.exe', ''))
                    except Exception as e:
                        failed_actions.append(f"Prozess {process}: {e}")

                if closed_processes:
                    actions_taken.append(f"Apps geschlossen: {', '.join(closed_processes)}")

            # 2. Fenster nach Titel schließen
            if self.security_actions['window_titles']:
                logger.info("Schließe Fenster nach Titel...")
                try:
                    if self.close_windows_by_title(self.security_actions['window_titles']):
                        actions_taken.append(
                            f"Fenster geschlossen: {len(self.security_actions['window_titles'])} Kategorien")
                except Exception as e:
                    failed_actions.append(f"Fenster schließen: {e}")

            # 3. Browser-Management
            logger.info("Browser-Management...")
            try:
                if self.close_browser_tabs_smart():
                    actions_taken.append("Browser-Management erfolgreich")
            except Exception as e:
                failed_actions.append(f"Browser-Management: {e}")

            # 4. System-Aktionen
            if self.security_actions['mute_system']:
                try:
                    if self.mute_system_simple():
                        actions_taken.append("System stumm geschaltet")
                except Exception as e:
                    failed_actions.append(f"Stummschaltung: {e}")

            if self.security_actions['minimize_all']:
                try:
                    if self.minimize_all_windows_simple():
                        actions_taken.append("Alle Fenster minimiert")
                except Exception as e:
                    failed_actions.append(f"Fenster minimieren: {e}")

            if self.security_actions['lock_screen']:
                try:
                    if self.lock_screen_simple():
                        actions_taken.append("Bildschirm gesperrt")
                except Exception as e:
                    failed_actions.append(f"Bildschirm sperren: {e}")

        except Exception as e:
            logger.error(f"Kritischer Fehler bei Sicherheitsmaßnahmen: {e}")
            failed_actions.append(f"Kritischer Fehler: {e}")

        # Ergebnis-Reporting
        if actions_taken:
            logger.info("✅ ERFOLGREICH AUSGEFÜHRT:")
            for action in actions_taken:
                logger.info(f"   • {action}")

        if failed_actions:
            logger.warning("❌ FEHLGESCHLAGEN:")
            for action in failed_actions:
                logger.warning(f"   • {action}")

        if not actions_taken and not failed_actions:
            logger.info("ℹ️  Keine Sicherheitsmaßnahmen konfiguriert")

        logger.info("=" * 60)

    def configure_security_settings(self):
        """Verbesserte GUI für Sicherheitseinstellungen"""
        try:
            settings_window = tk.Tk()
            settings_window.title("🎯 Sicherheitseinstellungen")
            settings_window.geometry("900x800")
            settings_window.configure(bg='white')

            # Haupttitel
            title_label = tk.Label(settings_window,
                                   text="🎯 Erweiterte Sicherheitseinstellungen",
                                   font=("Arial", 18, "bold"),
                                   bg='white', fg='darkblue')
            title_label.pack(pady=15)

            # Notebook für Tabs
            notebook = ttk.Notebook(settings_window)
            notebook.pack(fill='both', expand=True, padx=20, pady=10)

            # Tabs erstellen
            self._create_processes_tab(notebook)
            self._create_windows_tab(notebook)
            self._create_browser_tab(notebook)
            self._create_system_tab(notebook)

            # Listen initial laden
            self.update_process_list()
            self.update_title_list()

            # Haupt-Buttons
            button_frame = tk.Frame(settings_window, bg='white')
            button_frame.pack(fill='x', padx=20, pady=15)

            def save_all_settings():
                try:
                    # Browser-Einstellungen speichern
                    if hasattr(self, 'browser_vars'):
                        for action, var in self.browser_vars.items():
                            self.security_actions['browser_actions'][action] = var.get()

                    # System-Einstellungen speichern
                    if hasattr(self, 'system_vars'):
                        for action, var in self.system_vars.items():
                            self.security_actions[action] = var.get()

                    # Alle Einstellungen speichern
                    self.save_security_settings()
                    messagebox.showinfo("✅ Gespeichert", "Alle Einstellungen wurden erfolgreich gespeichert!")
                    settings_window.destroy()
                except Exception as e:
                    messagebox.showerror("❌ Fehler", f"Fehler beim Speichern: {e}")
                    logger.error(f"Fehler beim Speichern der Einstellungen: {e}")

            def test_all_security():
                try:
                    # Erst die aktuellen GUI-Einstellungen übernehmen
                    if hasattr(self, 'browser_vars'):
                        for action, var in self.browser_vars.items():
                            self.security_actions['browser_actions'][action] = var.get()

                    if hasattr(self, 'system_vars'):
                        for action, var in self.system_vars.items():
                            self.security_actions[action] = var.get()

                    messagebox.showinfo("🧪 Test", "Teste alle Sicherheitsmaßnahmen mit aktuellen Einstellungen...")

                    # Test in separatem Thread
                    test_thread = threading.Thread(target=self.execute_security_measures, daemon=True)
                    test_thread.start()

                except Exception as e:
                    messagebox.showerror("❌ Fehler", f"Fehler beim Testen: {e}")
                    logger.error(f"Fehler beim Testen: {e}")

            tk.Button(button_frame, text="💾 SPEICHERN", command=save_all_settings,
                      bg='green', fg='white', font=("Arial", 12, "bold"),
                      padx=20, pady=5).pack(side='left', padx=10)

            tk.Button(button_frame, text="🧪 TESTEN", command=test_all_security,
                      bg='orange', fg='white', font=("Arial", 12, "bold"),
                      padx=20, pady=5).pack(side='left', padx=10)

            tk.Button(button_frame, text="❌ ABBRECHEN", command=settings_window.destroy,
                      bg='red', fg='white', font=("Arial", 12, "bold"),
                      padx=20, pady=5).pack(side='left', padx=10)

            settings_window.mainloop()

        except Exception as e:
            logger.error(f"Fehler bei Einstellungs-GUI: {e}")
            messagebox.showerror("❌ Fehler", f"GUI-Fehler: {e}")

    def _create_processes_tab(self, notebook):
        """Erstellt Tab für Prozess-Management"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📱 Apps")

        # Titel
        tk.Label(frame, text="Apps die bei Eindringling geschlossen werden",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Schnellauswahl
        quick_frame = tk.Frame(frame)
        quick_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(quick_frame, text="Schnellauswahl:",
                 font=("Arial", 11, "bold")).pack(anchor='w')

        # Buttons für vordefinierte Kategorien
        for category, apps in self.predefined_apps.items():
            btn_frame = tk.Frame(quick_frame)
            btn_frame.pack(fill='x', pady=2)

            def add_category_apps(app_list=apps):
                for app in app_list:
                    if app not in self.security_actions['specific_processes']:
                        self.security_actions['specific_processes'].append(app)
                self.update_process_list()

            tk.Button(btn_frame, text=f"+ {category}",
                      command=add_category_apps,
                      bg='lightblue', font=("Arial", 9)).pack(side='left')

        # Liste der aktuellen Prozesse
        list_frame = tk.Frame(frame)
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)

        tk.Label(list_frame, text="Aktuelle Apps:",
                 font=("Arial", 11, "bold")).pack(anchor='w')

        # Listbox mit Scrollbar
        listbox_frame = tk.Frame(list_frame)
        listbox_frame.pack(fill='both', expand=True)

        self.process_listbox = Listbox(listbox_frame, height=10)
        self.process_listbox.pack(side='left', fill='both', expand=True)

        scrollbar = Scrollbar(listbox_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        self.process_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.process_listbox.yview)

        # Buttons
        btn_frame = tk.Frame(list_frame)
        btn_frame.pack(fill='x', pady=5)

        def add_custom_process():
            try:
                process = simpledialog.askstring("App hinzufügen",
                                                 "Prozessname (z.B. notepad.exe):")
                if process and process.strip() and process not in self.security_actions['specific_processes']:
                    self.security_actions['specific_processes'].append(process.strip())
                    self.update_process_list()
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des Prozesses: {e}")

        def remove_selected_process():
            try:
                selection = self.process_listbox.curselection()
                if selection:
                    process = self.process_listbox.get(selection[0])
                    self.security_actions['specific_processes'].remove(process)
                    self.update_process_list()
            except Exception as e:
                logger.error(f"Fehler beim Entfernen des Prozesses: {e}")

        tk.Button(btn_frame, text="➕ Hinzufügen", command=add_custom_process,
                  bg='green', fg='white').pack(side='left', padx=5)
        tk.Button(btn_frame, text="➖ Entfernen", command=remove_selected_process,
                  bg='red', fg='white').pack(side='left', padx=5)

        # Initial laden der Liste wird in configure_security_settings() gemacht

    def update_process_list(self):
        """Aktualisiert die Prozess-Liste"""
        if hasattr(self, 'process_listbox'):
            self.process_listbox.delete(0, tk.END)
            for process in self.security_actions['specific_processes']:
                self.process_listbox.insert(tk.END, process)

    def update_title_list(self):
        """Aktualisiert die Fenstertitel-Liste"""
        if hasattr(self, 'title_listbox'):
            self.title_listbox.delete(0, tk.END)
            for title in self.security_actions['window_titles']:
                self.title_listbox.insert(tk.END, title)

    def _create_windows_tab(self, notebook):
        """Erstellt Tab für Fenster-Management"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🪟 Fenster")

        tk.Label(frame, text="Fenster die nach Titel geschlossen werden",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Schnellauswahl für Fenstertitel
        quick_frame = tk.Frame(frame)
        quick_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(quick_frame, text="Schnellauswahl:",
                 font=("Arial", 11, "bold")).pack(anchor='w')

        # Buttons für vordefinierte Kategorien
        for category, titles in self.predefined_titles.items():
            btn_frame = tk.Frame(quick_frame)
            btn_frame.pack(fill='x', pady=2)

            def add_category_titles(title_list=titles):
                for title in title_list:
                    if title not in self.security_actions['window_titles']:
                        self.security_actions['window_titles'].append(title)
                self.update_title_list()

            tk.Button(btn_frame, text=f"+ {category}",
                      command=add_category_titles,
                      bg='lightgreen', font=("Arial", 9)).pack(side='left')

        # Liste der aktuellen Fenstertitel
        list_frame = tk.Frame(frame)
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)

        tk.Label(list_frame, text="Aktuelle Fenstertitel:",
                 font=("Arial", 11, "bold")).pack(anchor='w')

        # Listbox mit Scrollbar
        listbox_frame = tk.Frame(list_frame)
        listbox_frame.pack(fill='both', expand=True)

        self.title_listbox = Listbox(listbox_frame, height=10)
        self.title_listbox.pack(side='left', fill='both', expand=True)

        scrollbar2 = Scrollbar(listbox_frame, orient='vertical')
        scrollbar2.pack(side='right', fill='y')
        self.title_listbox.config(yscrollcommand=scrollbar2.set)
        scrollbar2.config(command=self.title_listbox.yview)

        # Buttons
        btn_frame = tk.Frame(list_frame)
        btn_frame.pack(fill='x', pady=5)

        def add_custom_title():
            try:
                title = simpledialog.askstring("Fenstertitel hinzufügen",
                                               "Titel oder Teilwort (z.B. YouTube):")
                if title and title.strip() and title not in self.security_actions['window_titles']:
                    self.security_actions['window_titles'].append(title.strip())
                    self.update_title_list()
            except Exception as e:
                logger.error(f"Fehler beim Hinzufügen des Titels: {e}")

        def remove_selected_title():
            try:
                selection = self.title_listbox.curselection()
                if selection:
                    title = self.title_listbox.get(selection[0])
                    self.security_actions['window_titles'].remove(title)
                    self.update_title_list()
            except Exception as e:
                logger.error(f"Fehler beim Entfernen des Titels: {e}")

        tk.Button(btn_frame, text="➕ Hinzufügen", command=add_custom_title,
                  bg='green', fg='white').pack(side='left', padx=5)
        tk.Button(btn_frame, text="➖ Entfernen", command=remove_selected_title,
                  bg='red', fg='white').pack(side='left', padx=5)

        # Initial laden der Liste wird in configure_security_settings() gemacht

    def _create_browser_tab(self, notebook):
        """Erstellt Tab für Browser-Management"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🌐 Browser")

        tk.Label(frame, text="Browser-spezifische Aktionen",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # Browser-Aktionen Checkboxen
        self.browser_vars = {}
        browser_actions = [
            ('close_youtube_tabs', '📺 YouTube Tabs schließen'),
            ('close_social_media_tabs', '📱 Social Media Tabs schließen'),
            ('close_streaming_tabs', '🎬 Streaming Tabs schließen'),
            ('close_all_browsers', '🚫 Alle Browser komplett schließen (DRASTISCH!)')
        ]

        for action, text in browser_actions:
            self.browser_vars[action] = BooleanVar(
                value=self.security_actions['browser_actions'].get(action, False)
            )

            cb = Checkbutton(frame, text=text,
                             variable=self.browser_vars[action],
                             font=("Arial", 11), bg='white')
            cb.pack(anchor='w', padx=20, pady=8)

        # Info-Text
        info_text = tk.Text(frame, height=6, wrap='word', font=("Arial", 9))
        info_text.pack(fill='x', padx=20, pady=10)
        info_text.insert('1.0',
                         "ℹ️ Browser-Aktionen Erklärung:\n\n"
                         "• YouTube/Social/Streaming Tabs: Versucht spezifische Tabs zu schließen\n"
                         "• Alle Browser schließen: Beendet alle Browser-Prozesse komplett\n\n"
                         "HINWEIS: Tab-spezifische Schließung ist experimentell und kann\n"
                         "unzuverlässig sein. Prozess-Schließung ist zuverlässiger aber drastischer.")
        info_text.config(state='disabled')

    def _create_system_tab(self, notebook):
        """Erstellt Tab für System-Aktionen"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="⚙️ System")

        tk.Label(frame, text="System-weite Aktionen",
                 font=("Arial", 14, "bold")).pack(pady=10)

        # System-Aktionen Checkboxen
        self.system_vars = {}
        system_actions = [
            ('minimize_all', '📦 Alle Fenster minimieren'),
            ('mute_system', '🔇 System stumm schalten'),
            ('lock_screen', '🔒 Bildschirm sperren')
        ]

        for action, text in system_actions:
            self.system_vars[action] = BooleanVar(
                value=self.security_actions.get(action, False)
            )

            cb = Checkbutton(frame, text=text,
                             variable=self.system_vars[action],
                             font=("Arial", 12), bg='white')
            cb.pack(anchor='w', padx=20, pady=12)

        # Plattform-Info
        platform_info = tk.Label(frame,
                                 text=f"Aktuelle Plattform: {platform.system()}",
                                 font=("Arial", 10), fg='gray')
        platform_info.pack(pady=10)

        # Verfügbarkeits-Status
        status_frame = tk.Frame(frame)
        status_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(status_frame, text="Verfügbare Funktionen:",
                 font=("Arial", 11, "bold")).pack(anchor='w')

        if PYAUTOGUI_AVAILABLE:
            tk.Label(status_frame, text="✅ PyAutoGUI verfügbar - Alle Funktionen aktiv",
                     font=("Arial", 10), fg='green').pack(anchor='w')
        else:
            tk.Label(status_frame, text="❌ PyAutoGUI nicht verfügbar - Installieren Sie: pip install pyautogui",
                     font=("Arial", 10), fg='red').pack(anchor='w')

        if WINDOWS_API_AVAILABLE:
            tk.Label(status_frame, text="✅ Windows API verfügbar",
                     font=("Arial", 10), fg='green').pack(anchor='w')
        elif platform.system() == "Windows":
            tk.Label(status_frame, text="⚠️ Windows API teilweise verfügbar",
                     font=("Arial", 10), fg='orange').pack(anchor='w')

    def show_intruder_popup(self):
        """Zeigt Eindringling-Popup mit verbesserter Thread-Sicherheit"""
        try:
            messages = [
                "🎯 EINDRINGLING ERKANNT!\nSicherheitsmaßnahmen werden aktiviert...",
                "⚠️ FREMDES GESICHT ERKANNT!\nSchutzprotokoll läuft...",
                "🕵️ UNBEKANNTE PERSON!\nSicherheitssystem aktiv..."
            ]

            import random
            message = random.choice(messages)

            # Sicherheitsmaßnahmen in separatem Thread
            security_thread = threading.Thread(target=self.execute_security_measures, daemon=True)
            security_thread.start()

            # Popup in Haupt-Thread
            try:
                messagebox.showwarning("🚨 SICHERHEITSALARM", message)
            except Exception as e:
                logger.error(f"Fehler bei Popup: {e}")

        except Exception as e:
            logger.error(f"Fehler bei Eindringling-Popup: {e}")

    def load_known_faces(self):
        """Lädt bekannte Gesichter mit verbesserter Fehlerbehandlung"""
        try:
            if os.path.exists("improved_faces.pkl"):
                with open("improved_faces.pkl", "rb") as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Geladene Personen: {list(self.known_faces.keys())}")
            else:
                logger.info("Keine gespeicherten Gesichter gefunden")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Gesichter: {e}")
            self.known_faces = {}

    def save_known_faces(self):
        """Speichert bekannte Gesichter"""
        try:
            with open("improved_faces.pkl", "wb") as f:
                pickle.dump(self.known_faces, f)
            logger.info("Gesichter erfolgreich gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Gesichter: {e}")

    def load_security_settings(self):
        """Lädt Sicherheitseinstellungen"""
        try:
            if os.path.exists("security_settings.pkl"):
                with open("security_settings.pkl", "rb") as f:
                    loaded_settings = pickle.load(f)
                    self.security_actions.update(loaded_settings)
                logger.info("Sicherheitseinstellungen geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Sicherheitseinstellungen: {e}")

    def save_security_settings(self):
        """Speichert Sicherheitseinstellungen"""
        try:
            with open("security_settings.pkl", "wb") as f:
                pickle.dump(self.security_actions, f)
            logger.info("Sicherheitseinstellungen gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Sicherheitseinstellungen: {e}")

    def capture_multiple_samples(self):
        """Verbesserte Trainingsbilder-Aufnahme"""
        try:
            name = None
            root = tk.Tk()
            root.withdraw()
            name = simpledialog.askstring("Name eingeben", "Wie heißt die Person?")
            root.destroy()

            if not name or not name.strip():
                return

            name = name.strip()
            logger.info(f"Erfasse Trainingsbilder für: {name}")

            if name not in self.known_faces:
                self.known_faces[name] = []

            samples_taken = 0
            samples_needed = 7  # Mehr Samples für bessere Erkennung

            while samples_taken < samples_needed:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.error("Fehler beim Lesen der Kamera")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    # Qualitätsprüfung des Gesichts
                    face_landmarks = results.multi_face_landmarks[0]
                    features = self.extract_enhanced_features(face_landmarks, frame.shape)

                    if features is not None:
                        status_text = f"Sample {samples_taken + 1}/{samples_needed} - SPACE für Aufnahme"
                        cv2.putText(frame, status_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Gesichtsrahmen zeichnen
                        h, w, _ = frame.shape
                        x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                        y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Gesichtsqualität zu niedrig!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Kein Gesicht erkannt!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Trainingsbilder aufnehmen - ESC zum Abbrechen', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and results.multi_face_landmarks:
                    features = self.extract_enhanced_features(results.multi_face_landmarks[0], frame.shape)
                    if features is not None:
                        self.known_faces[name].append(features)
                        samples_taken += 1
                        logger.info(f"Sample {samples_taken} für {name} aufgenommen")
                        time.sleep(0.8)  # Kurze Pause zwischen Aufnahmen
                elif key == 27:  # ESC
                    break

            cv2.destroyWindow('Trainingsbilder aufnehmen - ESC zum Abbrechen')

            if samples_taken > 0:
                self.save_known_faces()
                logger.info(f"{samples_taken} Trainingsbilder für {name} gespeichert")
                messagebox.showinfo("✅ Erfolgreich",
                                    f"{samples_taken} Trainingsbilder für {name} gespeichert!")
            else:
                logger.warning("Keine Trainingsbilder aufgenommen")

        except Exception as e:
            logger.error(f"Fehler bei Trainingsbilder-Aufnahme: {e}")
            messagebox.showerror("❌ Fehler", f"Fehler beim Training: {e}")

    def run(self):
        """Verbesserte Hauptschleife mit Fehlerbehandlung"""
        try:
            self.is_running = True
            logger.info("🎯 Verbesserte Face Security gestartet!")
            logger.info("=" * 70)
            logger.info("⌨️  Steuerung:")
            logger.info("   'q' = Beenden")
            logger.info("   'c' = Neues Gesicht trainieren")
            logger.info("   's' = Einstellungen anzeigen")
            logger.info("   'p' = Sicherheitseinstellungen")
            logger.info("   't' = Sicherheitsmaßnahmen testen")
            logger.info(f"   System: {platform.system()}")
            logger.info(f"   PyAutoGUI: {'✅ Verfügbar' if PYAUTOGUI_AVAILABLE else '❌ Nicht verfügbar'}")
            logger.info("=" * 70)

            # Gesichter-Training anbieten wenn keine vorhanden
            if not self.known_faces:
                response = messagebox.askyesno("Training",
                                               "Möchten Sie jetzt ein Gesicht trainieren?")
                if response:
                    self.capture_multiple_samples()

            logger.info("📹 Kamera aktiv - Überwachung läuft...")

            while self.is_running:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.error("Kamera-Fehler")
                    time.sleep(1)
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    results = self.face_mesh.process(rgb_frame)
                except Exception as e:
                    logger.error(f"MediaPipe Verarbeitungsfehler: {e}")
                    continue

                # Gesichtserkennung und -verarbeitung
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        try:
                            current_features = self.extract_enhanced_features(face_landmarks, frame.shape)
                            name, confidence = self.recognize_face(current_features)

                            # Sicherheitsprotokoll
                            if name == "EINDRINGLING" and time.time() > self.alert_cooldown:
                                self.show_intruder_popup()
                                self.alert_cooldown = time.time() + 20  # 20 Sekunden Cooldown

                            # Gesichtsrahmen zeichnen
                            self._draw_face_rectangle(frame, face_landmarks, name, confidence)

                        except Exception as e:
                            logger.error(f"Fehler bei Gesichtsverarbeitung: {e}")

                # Status-Informationen
                self._draw_status_info(frame)

                cv2.imshow('🎯 Verbesserte Face Security', frame)

                # Tastatur-Eingaben
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
                    logger.info("Teste Sicherheitsmaßnahmen...")
                    self.execute_security_measures()

        except KeyboardInterrupt:
            logger.info("Programm durch Benutzer beendet")
        except Exception as e:
            logger.error(f"Kritischer Fehler in Hauptschleife: {e}")
            messagebox.showerror("❌ Kritischer Fehler", f"Unerwarteter Fehler: {e}")
        finally:
            self.cleanup()

    def _draw_face_rectangle(self, frame, face_landmarks, name, confidence):
        """Zeichnet Gesichtsrahmen mit Informationen"""
        try:
            color = (0, 255, 0) if name != "EINDRINGLING" else (0, 0, 255)
            h, w, _ = frame.shape

            # Gesichtskoordinaten berechnen
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Rahmen und Hintergrund
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max + 70), color, 2)
            cv2.rectangle(frame, (x_min, y_max), (x_max, y_max + 70), color, cv2.FILLED)

            # Text
            cv2.putText(frame, name, (x_min + 5, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if name != "EINDRINGLING":
                confidence_text = f"Vertrauen: {confidence:.1%}"
                cv2.putText(frame, confidence_text, (x_min + 5, y_max + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            logger.error(f"Fehler beim Zeichnen des Gesichtsrahmens: {e}")

    def _draw_status_info(self, frame):
        """Zeichnet Status-Informationen"""
        try:
            total_people = len(self.known_faces)
            total_processes = len(self.security_actions['specific_processes'])
            total_titles = len(self.security_actions['window_titles'])

            status = f"Personen: {total_people} | Apps: {total_processes} | Fenster: {total_titles}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            security_status = f"Sicherheit: AKTIV | {platform.system()}"
            if PYAUTOGUI_AVAILABLE:
                security_status += " | PyAutoGUI: OK"
            cv2.putText(frame, security_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        except Exception as e:
            logger.error(f"Fehler beim Zeichnen der Status-Info: {e}")

    def show_settings(self):
        """Zeigt detaillierte Einstellungen in der Konsole"""
        try:
            logger.info("\n" + "=" * 70)
            logger.info("🎯 AKTUELLE EINSTELLUNGEN")
            logger.info("=" * 70)

            logger.info(f"👥 Bekannte Personen: {len(self.known_faces)}")
            for name, samples in self.known_faces.items():
                logger.info(f"   - {name}: {len(samples)} Trainingsbilder")

            logger.info(f"\n📱 Apps ({len(self.security_actions['specific_processes'])}):")
            for process in self.security_actions['specific_processes']:
                logger.info(f"   🚫 {process}")

            logger.info(f"\n🪟 Fenstertitel ({len(self.security_actions['window_titles'])}):")
            for title in self.security_actions['window_titles']:
                logger.info(f"   🚫 {title}")

            logger.info("\n🌐 Browser-Aktionen:")
            for action, enabled in self.security_actions['browser_actions'].items():
                status = "✅ AKTIV" if enabled else "❌ DEAKTIVIERT"
                action_name = action.replace('_', ' ').title()
                logger.info(f"   {action_name}: {status}")

            logger.info("\n⚙️ System-Aktionen:")
            system_actions = ['minimize_all', 'mute_system', 'lock_screen']
            for action in system_actions:
                enabled = self.security_actions.get(action, False)
                status = "✅ AKTIV" if enabled else "❌ DEAKTIVIERT"
                action_name = action.replace('_', ' ').title()
                logger.info(f"   {action_name}: {status}")

            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Fehler beim Anzeigen der Einstellungen: {e}")

    def cleanup(self):
        """Aufräumarbeiten beim Beenden"""
        try:
            self.is_running = False
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup abgeschlossen")
        except Exception as e:
            logger.error(f"Fehler beim Cleanup: {e}")


def main():
    """Hauptfunktion mit verbesserter Fehlerbehandlung"""
    try:
        # Abhängigkeiten prüfen
        missing_modules = []

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            missing_modules.append("scikit-learn")

        try:
            import psutil
        except ImportError:
            missing_modules.append("psutil")

        if missing_modules:
            logger.error("❌ Fehlende Abhängigkeiten:")
            for module in missing_modules:
                logger.error(f"   📦 pip install {module}")
            sys.exit(1)

        # Warnung bei fehlenden optionalen Modulen
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("⚠️  PyAutoGUI nicht verfügbar:")
            logger.warning("   📦 pip install pyautogui")
            logger.warning("   💡 Einige erweiterte Funktionen sind deaktiviert")
            time.sleep(2)

        # Anwendung starten
        app = ImprovedFaceSecurity()
        app.run()

    except Exception as e:
        logger.error(f"Kritischer Fehler beim Start: {e}")
        messagebox.showerror("❌ Kritischer Fehler",
                             f"Anwendung konnte nicht gestartet werden:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()