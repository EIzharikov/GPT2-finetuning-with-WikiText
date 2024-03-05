"""
Checks dependencies
"""
import re
import sys
from pathlib import Path

from config.constants import PROJECT_ROOT


def get_paths() -> list:
    """
    Returns list of paths to non-python files
    """
    list_with_paths = []
    for file in PROJECT_ROOT.iterdir():
        if file.name in ['requirements.txt', 'requirements_ci.txt']:
            list_with_paths.append(file)
    return list_with_paths


def get_requirements(path: Path) -> list:
    """
    Returns a list of dependencies
    """
    with open(path, 'r', encoding='utf-8') as requirements_file:
        lines = requirements_file.readlines()
    return [line.strip() for line in lines if line.strip()]


def compile_pattern() -> re.Pattern:
    """
    Returns the compiled pattern
    """
    return re.compile(r'\w+(-\w+|\[\w+\])*==\d+(\.\d+)+')


def check_dependencies(lines: list, compiled_pattern: re.Pattern, path:Path) -> bool:
    """
    Checks that dependencies confirm to the template
    """
    if sorted(lines) != lines:
        print(f'Dependencies in {path} do not conform to the template.')
        return False
    for line in lines:
        if not re.search(compiled_pattern, line):
            print(f'Dependencies in {path}  do not conform to the template.')
            return False
    print(f'Dependencies in {path}: OK.')
    return True


def main() -> None:
    """
    Calls functions
    """
    paths = get_paths()
    compiled_pattern = compile_pattern()
    requirements_status = True
    for path in paths:
        lines = get_requirements(path)
        requirements_status = check_dependencies(lines, compiled_pattern, path)
    sys.exit(not requirements_status)


if __name__ == '__main__':
    main()
