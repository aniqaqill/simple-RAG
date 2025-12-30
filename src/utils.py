import os
from typing import List

def load_text_file(file_path: str) -> List[str]:
    """
    Loads a text file and returns a list of non-empty lines.
    
    Args:
        file_path: Path to the text file.
        
    Returns:
        List[str]: List of lines.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]
