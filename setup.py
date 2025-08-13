import subprocess
import sys

def install_requirements():
    """Install requirements with specific versions"""
    requirements = [
        "streamlit>=1.28.0,<1.32.0",
        "tensorflow>=2.15.0,<2.17.0", 
        "keras>=2.15.0,<2.17.0",
        "huggingface_hub>=0.17.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0,<1.25.0",
        "pandas>=1.5.0",
        "h5py>=3.7.0"
    ]
    
    for req in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])

if __name__ == "__main__":
    install_requirements()
