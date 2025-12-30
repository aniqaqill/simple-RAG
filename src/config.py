import os
import subprocess

def get_windows_host_ip() -> str:
    """Automatically finds the Windows host IP from inside WSL."""
    try:
        # Grabs the gateway IP (your Windows side)
        cmd = "ip route | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    except Exception:
        return "127.0.0.1" # Fallback

# Network Configuration
WINDOWS_HOST_IP = get_windows_host_ip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{WINDOWS_HOST_IP}:11434")

# Model Configuration
# Note: Ensure you have run 'ollama pull <model_name>' on Windows first!
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
