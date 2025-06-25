import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import os
import threading
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class ImprovedFaceAlert:
    def __init__(self):
        # MediaPipe Setup
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Erhöhte Empfindlichkeit
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)  # Niedrigere Schwelle
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=5,
            refine_landmarks=True, min_detection_confidence=0.5)

        # Face Recognition Setup - Mehrere Samples pro Person
        self.known_faces = {}  # Dict: name -> [liste von features]
        self.video_capture = cv2.VideoCapture(0)
        self.alert_cooldown = 0

        # Verbesserte Parameter
        self.recognition_threshold = 0.85  # Cosine Similarity Schwelle
        self.min_samples_per_person = 3  # Mindestens 3 Samples pro Person

        self.load_known_faces()

    def extract_enhanced_features(self, landmarks, frame_shape):
        """Verbesserte Feature-Extraktion mit mehr Details"""
        if not landmarks:
            return None

        h, w = frame_shape[:2]

        # Alle wichtigen Gesichtspunkte (mehr als vorher)
        key_points = [
            # Gesichtskontur
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            # Augenbrauen
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            # Augen (links und rechts)
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Nase
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321,
            308, 324, 318,
            # Mund
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            # Kinn
            18, 175, 199, 175, 199, 175, 199
        ]

        # Features extrahieren
        features = []

        # 1. Koordinaten der Schlüsselpunkte
        for point_idx in key_points:
            if point_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[point_idx]
                features.extend([landmark.x, landmark.y, landmark.z])

        # 2. Geometrische Verhältnisse berechnen
        if len(landmarks.landmark) > 300:
            # Augenabstand
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[362]
            eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)

            # Nase zu Mund Verhältnis
            nose_tip = landmarks.landmark[1]
            mouth_center = landmarks.landmark[13]
            nose_mouth_dist = np.sqrt((nose_tip.x - mouth_center.x) ** 2 + (nose_tip.y - mouth_center.y) ** 2)

            # Gesichtshöhe vs Breite
            top = landmarks.landmark[10]
            bottom = landmarks.landmark[175]
            left = landmarks.landmark[234]
            right = landmarks.landmark[454]

            face_height = abs(top.y - bottom.y)
            face_width = abs(left.x - right.x)
            face_ratio = face_height / (face_width + 0.001)

            # Geometrische Features hinzufügen
            features.extend([eye_distance, nose_mouth_dist, face_ratio])

        return np.array(features)

    def compare_faces_improved(self, features1, features2):
        """Verbesserte Gesichtserkennung mit Cosine Similarity"""
        if features1 is None or features2 is None:
            return 0.0

        # Sicherstellen dass beide gleiche Länge haben
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]

        # Cosine Similarity berechnen (besser als Euklidische Distanz)
        similarity = cosine_similarity([features1], [features2])[0][0]
        return similarity

    def recognize_face(self, current_features):
        """Erkennt Gesicht gegen alle bekannten Personen"""
        if current_features is None:
            return "EINDRINGLING", 0.0

        best_match = "EINDRINGLING"
        best_score = 0.0

        for name, face_samples in self.known_faces.items():
            # Vergleiche mit allen Samples dieser Person
            similarities = []
            for sample_features in face_samples:
                sim = self.compare_faces_improved(current_features, sample_features)
                similarities.append(sim)

            # Nimm den besten Match von allen Samples
            if similarities:
                max_similarity = max(similarities)
                avg_similarity = np.mean(similarities)

                # Gewichteter Score (70% bester Match, 30% Durchschnitt)
                final_score = 0.7 * max_similarity + 0.3 * avg_similarity

                if final_score > best_score and final_score > self.recognition_threshold:
                    best_match = name
                    best_score = final_score

        return best_match, best_score

    def load_known_faces(self):
        """Lädt bekannte Gesichter"""
        if os.path.exists("improved_faces.pkl"):
            try:
                with open("improved_faces.pkl", "rb") as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Geladene Personen: {list(self.known_faces.keys())}")
                for name, samples in self.known_faces.items():
                    print(f"   {name}: {len(samples)} Trainingsbilder")
            except Exception as e:
                print(f"❌ Fehler beim Laden: {e}")

        if not self.known_faces:
            print("🤷 Keine bekannten Gesichter gefunden!")

    def save_known_faces(self):
        """Speichert bekannte Gesichter"""
        with open("improved_faces.pkl", "wb") as f:
            pickle.dump(self.known_faces, f)
        print("💾 Gesichter gespeichert!")

    def show_intruder_popup(self):
        """Einfaches Pop-up ohne Threading (funktioniert perfekt!)"""
        messages = [
            "🚨 EINDRINGLING ERKANNT! 🚨\nWer bist du denn?!",
            "⚠️ FREMDES GESICHT! ⚠️\nZeig deinen Ausweis!",
            "🕵️ UNBEKANNTE PERSON! 🕵️\nDu kommst hier nicht rein!",
            "🚫 ZUGANG VERWEIGERT! 🚫\nNur VIPs erlaubt!",
            "👽 ALIEN DETECTED! 👽\nBist du von einem anderen Planeten?",
            "🤖 BEEP BEEP! 🤖\nRoboter-Sicherheitssystem aktiviert!",
            "🎭 MASKENBALL? 🎭\nOder bist du einfach neu hier?",
            "🔍 GESICHT UNBEKANNT! 🔍\nBist du ein Spion?!"
        ]

        import random
        message = random.choice(messages)

        # Pop-up direkt im Hauptthread - kein Threading nötig!
        messagebox.showwarning("🚨 SICHERHEITSALARM 🚨", message)

    def capture_multiple_samples(self):
        """Erfasst mehrere Trainingsbilder einer Person"""
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
        print("❌ ESC zum Abbrechen")

        if name not in self.known_faces:
            self.known_faces[name] = []

        samples_needed = max(0, self.min_samples_per_person - len(self.known_faces[name]))
        samples_taken = 0

        while samples_taken < samples_needed + 2:  # 2 extra Samples
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Gesichter erkennen
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Landmarks zeichnen
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

                status_text = f"Sample {samples_taken + 1} - Drücke SPACE"
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Für {name}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Kein Gesicht erkannt!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Trainingsbilder aufnehmen', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space
                if results.multi_face_landmarks:
                    # Features extrahieren
                    features = self.extract_enhanced_features(
                        results.multi_face_landmarks[0], frame.shape)
                    if features is not None:
                        self.known_faces[name].append(features)
                        samples_taken += 1
                        print(f"✅ Sample {samples_taken} für {name} gespeichert!")

                        if samples_taken >= samples_needed + 2:
                            break

                        time.sleep(0.5)  # Kurze Pause
                else:
                    print("❌ Kein Gesicht erkannt! Versuche es nochmal.")
            elif key == 27:  # ESC
                break

        cv2.destroyWindow('Trainingsbilder aufnehmen')

        if samples_taken > 0:
            self.save_known_faces()
            print(f"🎉 {samples_taken} Trainingsbilder für {name} gespeichert!")
            print(f"📊 {name} hat jetzt {len(self.known_faces[name])} Trainingsbilder")
        else:
            print("❌ Keine Trainingsbilder aufgenommen!")

    def run(self):
        """Hauptschleife mit verbesserter Erkennung"""
        print("🚀 Verbesserte Face Recognition Alert gestartet!")
        print("=" * 60)
        print("⌨️  Steuerung:")
        print("   'q' = Beenden")
        print("   'c' = Neues Gesicht trainieren")
        print("   's' = Einstellungen anzeigen")
        print("=" * 60)

        if not self.known_faces:
            response = input("🤔 Möchtest du dein Gesicht jetzt trainieren? (j/n): ")
            if response.lower() == 'j':
                self.capture_multiple_samples()

        print("\n📹 Kamera gestartet... Viel Spaß!")

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Gesichter erkennen
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Features extrahieren
                    current_features = self.extract_enhanced_features(face_landmarks, frame.shape)

                    # Gesicht erkennen
                    name, confidence = self.recognize_face(current_features)

                    # Pop-up für Eindringlinge
                    if name == "EINDRINGLING" and time.time() > self.alert_cooldown:
                        self.show_intruder_popup()
                        self.alert_cooldown = time.time() + 5

                    # Gesicht zeichnen
                    color = (0, 255, 0) if name != "EINDRINGLING" else (0, 0, 255)

                    # Bounding Box
                    h, w, _ = frame.shape
                    x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                    y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Rechteck und Info zeichnen
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max + 60), color, 2)
                    cv2.rectangle(frame, (x_min, y_max), (x_max, y_max + 60), color, cv2.FILLED)

                    # Name und Konfidenz anzeigen
                    cv2.putText(frame, name, (x_min + 5, y_max + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if name != "EINDRINGLING":
                        confidence_text = f"{confidence:.1%}"
                        cv2.putText(frame, confidence_text, (x_min + 5, y_max + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Status anzeigen
            total_people = len(self.known_faces)
            total_samples = sum(len(samples) for samples in self.known_faces.values())
            status = f"Personen: {total_people} | Trainingsbilder: {total_samples}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Schwelle anzeigen
            threshold_text = f"Schwelle: {self.recognition_threshold:.0%}"
            cv2.putText(frame, threshold_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('🚀 Verbesserte Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.capture_multiple_samples()
            elif key == ord('s'):
                self.show_settings()

        self.video_capture.release()
        cv2.destroyAllWindows()

    def show_settings(self):
        """Zeigt aktuelle Einstellungen"""
        print("\n" + "=" * 50)
        print("⚙️  AKTUELLE EINSTELLUNGEN")
        print("=" * 50)
        print(f"🎯 Erkennungsschwelle: {self.recognition_threshold:.0%}")
        print(f"📸 Min. Trainingsbilder: {self.min_samples_per_person}")
        print(f"👥 Bekannte Personen: {len(self.known_faces)}")
        for name, samples in self.known_faces.items():
            print(f"   - {name}: {len(samples)} Trainingsbilder")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # Sklearn für Cosine Similarity installieren falls nötig
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("❌ sklearn nicht gefunden!")
        print("📦 Installiere: pip install scikit-learn")
        exit(1)

    app = ImprovedFaceAlert()
    app.run()