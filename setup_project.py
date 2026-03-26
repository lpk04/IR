import os

# =========================
# ROOT DIRECTORY
# =========================
ROOT = r"D:\IR\demo"

# =========================
# FOLDER STRUCTURE
# =========================
folders = [
    "data",
    "src",
    "app",
    "models",
    "trace",
]

# =========================
# FILE STRUCTURE
# =========================
files = [
    "main.py",
    "requirements.txt",
    "README.md",
    
    # src
 
    # app
    "app/app.py",

    # notebooks
    "notebooks/demo.ipynb"
]

# =========================
# CREATE FOLDERS
# =========================
for folder in folders:
    path = os.path.join(ROOT, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created folder: {path}")

# =========================
# CREATE FILES
# =========================
for file in files:
    path = os.path.join(ROOT, file)

    # tạo folder cha nếu chưa có
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # tạo file rỗng
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass

    print(f"Created file: {path}")

print("\n✅ Project structure created successfully!")