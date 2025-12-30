import pytest
import os
from src.utils import load_text_file

def test_load_text_file_success(tmp_path):
    # Create a temporary file
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "test.txt"
    p.write_text("line1\n\nline2\n", encoding="utf-8")
    
    result = load_text_file(str(p))
    assert len(result) == 2
    assert result[0] == "line1"
    assert result[1] == "line2"

def test_load_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_text_file("non_existent_file.txt")
