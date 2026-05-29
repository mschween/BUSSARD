import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
source_dir = Path("data/mit-67")
target_dir = Path("data/balanced_mit-67")

# Get all files from both scenes
dining_files = list((source_dir / "dining_room").glob("*"))
office_files = list((source_dir / "office").glob("*"))

print(f"Found {len(dining_files)} dining_room samples")
print(f"Found {len(office_files)} office samples")

# Determine balanced size (minimum of both)
balanced_size = min(len(dining_files), len(office_files))
print(f"\nCreating balanced dataset with {balanced_size} samples per scene")

# Randomly sample from the larger set
random.shuffle(dining_files)
random.shuffle(office_files)

selected_dining = dining_files[:balanced_size]
selected_office = office_files[:balanced_size]

# Create target directory structure
(target_dir / "dining_room").mkdir(parents=True, exist_ok=True)
(target_dir / "office").mkdir(parents=True, exist_ok=True)

# Create symlinks
for f in selected_dining:
    target = target_dir / "dining_room" / f.name
    target.symlink_to(f.resolve())

for f in selected_office:
    target = target_dir / "office" / f.name
    target.symlink_to(f.resolve())

print(f"\nBalanced dataset created in {target_dir}")
print(f"Total samples: {2 * balanced_size}")
