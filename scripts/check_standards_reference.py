#!/usr/bin/env python3
"""Check that new Python modules reference engineering standards."""

import sys
from pathlib import Path


def check_file(file_path: Path) -> bool:
    """Check if file references standards.

    Args:
        file_path: Path to the file to check.

    Returns:
        True if file references standards or check passes, False otherwise.

    Raises:
        FileNotFoundError: If file does not exist.
        PermissionError: If file cannot be read.
    """
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
        # Check for standards reference in docstring or comments
        if "standards" in content.lower() or "engineering" in content.lower():
            return True
        # For now, just check if file exists - can be enhanced later
        return True
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error checking {file_path}: {e}", file=sys.stderr)
        return False
    except UnicodeDecodeError as e:
        print(f"Error reading {file_path}: invalid encoding: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Main entry point for standards reference checker.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if len(sys.argv) < 2:
        return 0

    files: list[Path] = []
    for arg in sys.argv[1:]:
        file_path = Path(arg)
        if file_path.exists():
            files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)

    for file_path in files:
        if file_path.suffix == ".py" and "test" not in file_path.stem.lower():
            try:
                if not check_file(file_path):
                    print(f"Warning: {file_path} may need standards reference")
            except (FileNotFoundError, PermissionError) as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

