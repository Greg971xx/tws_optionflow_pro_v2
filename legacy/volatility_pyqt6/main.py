import os, sys
# 0) Flags env
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-gpu-compositing --no-sandbox --disable-features=RendererCodeIntegrity")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"

# 1) Attributs Qt AVANT QApplication
from PyQt6.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

# 2) ⚠️ “Amorce” WebEngine AVANT QApplication (si non SAFE_MODE)
if not SAFE_MODE:
    try:
        import PyQt6.QtWebEngineWidgets  # amorce correcte AVANT QApplication
    except Exception as e:
        print(f"⚠️ WebEngine non initialisé (amorce): {e}")

# 3) Créer QApplication
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv if sys.argv else ["tws_optionflow"])

# 4) (Optionnel) Configurer les profils/cache APRÈS QApplication
if not SAFE_MODE:
    try:
        from PyQt6.QtWebEngineCore import QWebEngineProfile
        os.makedirs(r"C:\Temp\qtweb_profile", exist_ok=True)
        os.makedirs(r"C:\Temp\qtweb_cache", exist_ok=True)
        prof = QWebEngineProfile.defaultProfile()
        prof.setPersistentStoragePath(r"C:\Temp\qtweb_profile")
        prof.setCachePath(r"C:\Temp\qtweb_cache")
    except Exception as e:
        print(f"⚠️ WebEngine profile skipped: {e}")

# 5) Importer l’UI
from ui.main_window import MainWindow
win = MainWindow()
win.show()
sys.exit(app.exec())
