import os

# =========================
# ROOT DIRECTORY
# =========================
ROOT = r"D:\IR\demo"

# =========================
# FOLDER STRUCTURE
# =========================
folders = [
    "app",
    "data",
    "data/processed",
    "index",
    "models",
    "notebooks",
    "results",
    "run",
    "src",
    "trace",
]

# =========================
# FILE STRUCTURE
# =========================
files = [
    "main.py",
    "requirements.txt",
    "README.md",

    # app
    "app/app.py",

    # notebooks
    "notebooks/demo.ipynb",

    # src core
    "src/config.py",
]

# =========================
# CREATE FOLDERS
# =========================
print("📁 Creating folders...\n")

for folder in folders:
    path = os.path.join(ROOT, folder)

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✅ Created folder: {path}")
    else:
        print(f"⚠️ Already exists: {path}")

# =========================
# CREATE FILES
# =========================
print("\n📄 Creating files...\n")

for file in files:
    path = os.path.join(ROOT, file)

    # tạo folder cha nếu chưa có
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # tạo file nếu chưa tồn tại
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            pass
        print(f"✅ Created file: {path}")
    else:
        print(f"⚠️ Already exists: {path}")

print("\n🚀 Project setup completed!")