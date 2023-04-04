import sys
from pathlib import Path


def get_repo_root():
    cwd = Path.cwd()
    cwd_files = [path.name for path in cwd.iterdir()]
    while '.git' not in cwd_files:
        cwd = cwd.parent
        if cwd == Path('/'):
            raise RuntimeError('_init_paths: project not a git repository (or any of the parent directories)')
        cwd_files = [path.name for path in cwd.iterdir()]
    return cwd

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

repo_root = get_repo_root()

def _init_path(p_rel, recursive=False):
    if not isinstance(p_rel, Path):
        p_rel = Path(p_rel)
        
    p_abs = repo_root.joinpath(p_rel)
    add_path(str(p_abs))

    if recursive:
        for p in sorted(p_abs.parents, reverse=True):
            if p == repo_root:
                break
            add_path(str(p))
    return