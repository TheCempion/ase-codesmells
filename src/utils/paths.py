from pathlib import Path


__all__ = [
    'add_suffix',
    'get_safe_path',
    'get_safe_filename',
]


def get_safe_filename(path: Path) -> Path:
    filename, ext = path.stem, path.suffix
    parent_path = path.parent
    parent_path.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return path

    get_new_path = lambda n: parent_path / f'{filename}_{n}{ext}'
    file_num = 1
    while get_new_path(file_num).exists():
        file_num += 1
    return get_new_path(file_num)


def get_safe_path(path: Path, create: bool = True, must_have_number: bool = False) -> Path:
    get_new_path = lambda n: add_suffix(path, f'_{n}')
    if path.exists() or must_have_number:
        file_num = 1
        while get_new_path(file_num).exists():
            file_num += 1
        new_path = get_new_path(file_num) 
    else:
        new_path = path

    if create:
        new_path.mkdir(parents=True)
    return new_path


def add_suffix(path: Path, suffix: str, before_extension: bool = False) -> Path:
    filename, ext = path.stem, path.suffix
    if before_extension:
        new_filename = filename + suffix + ext
    else:
        new_filename = filename + ext + suffix 
    return path.parent / new_filename
