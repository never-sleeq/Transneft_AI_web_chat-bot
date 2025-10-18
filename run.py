import os
import sys
import subprocess
import threading
import time
import webbrowser

BACKEND_DIR = "transneft_ai_web_chat/backend"
FRONTEND_DIR = "../frontend"

def run_backend():
    print("Запуск бэкенда на http://localhost:8001...")
    os.chdir(BACKEND_DIR)
    subprocess.run([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", "0.0.0.0",
        "--port", "8001"
    ])

def run_frontend():
    print("Запуск фронтенда на http://localhost:8005...")
    os.chdir(FRONTEND_DIR)
    subprocess.run([sys.executable, "-m", "http.server", "8005"])

def main():
    print("\nЗапуск системы 'Цифровой консультант Транснефть'")

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    time.sleep(3)
    webbrowser.open("http://localhost:8005")
    run_frontend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nСистема остановлена.")
    except Exception as e:
        print(f"Ошибка: {e}")