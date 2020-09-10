import os
import pathlib

INPUT_FILES_PATH = pathlib.Path(__file__).parent / 'input_files'


class GeneratedTest:
    generate: bool = False

    def generate_or_check(self, content: bytes, file_name: str):
        file_path = INPUT_FILES_PATH / file_name
        if self.generate:
            file_path.write_bytes(content)
        else:
            check_content = file_path.read_bytes()
            assert content == check_content
