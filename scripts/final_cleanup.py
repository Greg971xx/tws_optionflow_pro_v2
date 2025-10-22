"""
Final cleanup - Remove all obsolete files
"""
import shutil
from pathlib import Path
from datetime import datetime


def final_cleanup():
    """Remove obsolete files and folders"""

    root = Path(".")
    archive_dir = root / f"_archive_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(exist_ok=True)

    print("🧹 Final Project Cleanup")
    print("=" * 70)

    # Items to archive
    obsolete_items = [
        "core",  # Old core folder (not src/core)
        "ui",  # If exists
        "ui",  # If exists (old ui, not src/ui)
        "utils",  # If exists
        "tools",  # If exists
        "legacy",  # Already archived but double-check
    ]

    archived_count = 0

    for item_name in obsolete_items:
        item_path = root / item_name

        if item_path.exists() and item_path != root / "src":
            print(f"\n📦 Archiving: {item_name}")

            if item_path.is_dir():
                # Archive folder
                shutil.make_archive(
                    str(archive_dir / item_name),
                    'zip',
                    str(item_path)
                )

                # Delete original
                shutil.rmtree(item_path)
                print(f"   ✓ Archived and removed")
                archived_count += 1
            else:
                # Archive file
                shutil.copy(item_path, archive_dir / item_name)
                item_path.unlink()
                print(f"   ✓ Archived and removed")
                archived_count += 1

    # Clean root Python files
    print("\n📄 Checking root Python files...")

    root_py_files = list(root.glob("*.py"))
    keep_files = {"main.py", "setup.py", "config.py"}

    for py_file in root_py_files:
        if py_file.name not in keep_files:
            if any(pattern in py_file.name for pattern in ["old", "test", "temp", "backup"]):
                print(f"   Moving: {py_file.name}")
                shutil.move(str(py_file), archive_dir / py_file.name)
                archived_count += 1

    # Clean __pycache__ recursively
    print("\n🗑️  Removing __pycache__ folders...")
    pycache_count = 0
    for pycache in root.rglob("__pycache__"):
        shutil.rmtree(pycache)
        pycache_count += 1

    if pycache_count > 0:
        print(f"   ✓ Removed {pycache_count} __pycache__ folder(s)")

    # Summary
    print("\n" + "=" * 70)
    print("✅ CLEANUP COMPLETE")
    print("=" * 70)

    if archived_count > 0:
        print(f"\n📦 Archived {archived_count} item(s) to:")
        print(f"   {archive_dir}")
    else:
        print("\n✨ No obsolete items found - project already clean!")

    print("\n📁 Final structure:")
    print_tree(root, max_depth=2)


def print_tree(directory: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0):
    """Print directory tree"""
    if current_depth >= max_depth:
        return

    skip_dirs = {'.git', '__pycache__', '.idea', '.vscode', 'venv', '.venv', 'node_modules'}
    skip_patterns = {'_archive', '.db', '.log'}

    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    except PermissionError:
        return

    # Filter items
    items = [
        item for item in items
        if not any(skip in item.name for skip in skip_patterns)
           and item.name not in skip_dirs
           and not item.name.startswith('.')
    ]

    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "

        print(f"{prefix}{current_prefix}{item.name}")

        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1)


if __name__ == "__main__":
    print("\n⚠️  This will archive and remove obsolete files.")
    print("   A backup will be created automatically.")

    response = input("\n   Continue? (yes/no): ").strip().lower()

    if response == 'yes':
        final_cleanup()
    else:
        print("\n❌ Cleanup cancelled.")