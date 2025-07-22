import sys
import subprocess
import importlib
from packaging import version

def check_and_install_tensorflow():
    """Check TensorFlow installation and handle dependencies"""
    try:
        import tensorflow as tf
        print(f"TensorFlow {tf.__version__} is already installed")
        return True
    except ImportError:
        print("TensorFlow not found. Installing...")
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "tensorflow-gpu==2.10.0", 
                "typing-extensions>=4.6.0",
                "packaging",  # Ensure version comparison package is available
                "--use-deprecated=legacy-resolver"
            ])
            importlib.invalidate_caches()
            import tensorflow as tf
            print(f"Successfully installed TensorFlow {tf.__version__}")
            return True
        except Exception as e:
            print(f"Installation failed: {e}")
            return False

def verify_gpu_support():
    """Verify GPU availability and dependencies"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("GPU devices found:")
            for gpu in gpus:
                print(f"- {gpu.name}")
            return True
        else:
            print("No GPU devices found. Falling back to CPU.")
            return False
    except Exception as e:
        print(f"GPU verification failed: {e}")
        return False

def check_typing_extensions():
    """Check typing-extensions version properly"""
    try:
        import typing_extensions
        # Get version through package metadata
        from importlib.metadata import version as get_package_version
        ext_version = get_package_version('typing-extensions')
        
        if version.parse(ext_version) < version.parse("4.6.0"):
            print(f"typing-extensions {ext_version} is too old. Upgrading...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--upgrade", 
                "typing-extensions>=4.6.0"
            ])
            return True
        print(f"typing-extensions {ext_version} is compatible")
        return True
    except Exception as e:
        print(f"Error checking typing-extensions: {e}")
        return False

def setup_environment():
    """Main setup function"""
    # Step 1: Check/install TensorFlow
    if not check_and_install_tensorflow():
        print("Failed to set up TensorFlow")
        return False
    
    # Step 2: Verify GPU support
    if not verify_gpu_support():
        print("Continuing with CPU support only")
    
    # Step 3: Check typing-extensions using proper version detection
    if not check_typing_extensions():
        print("Warning: Could not verify typing-extensions version")
    
    return True

if __name__ == "__main__":
    if setup_environment():
        print("\nEnvironment setup completed successfully!")
        print("You can now run your TensorFlow code.")
    else:
        print("\nEnvironment setup failed. Please check the error messages.")
        print("Possible solutions:")
        print("1. Update your NVIDIA drivers")
        print("2. Install CUDA/cuDNN manually")
        print("3. Try CPU-only version: pip install tensorflow-cpu")