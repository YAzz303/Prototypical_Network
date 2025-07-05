import os
import shutil
from tqdm import tqdm

def reorganize_dataset(src_dir, dst_root):
    """
    Reorganizes a single folder dataset into train/test split based on script prefixes
    Args:
        src_dir: Path to your current dataset (with all characters mixed)
        dst_root: Path where new organized dataset will be created
    """
    # Create target directories
    os.makedirs(os.path.join(dst_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'test'), exist_ok=True)
    
    # Define which prefixes go to train vs test
    train_prefixes = ['bhujimol', 'brahmi', 'kirat', 'lichhavi','nandanagiri']
    test_prefixes = ['maithili', 'tibetan']  # Add your test script prefixes here
    
    for char_dir in tqdm(os.listdir(src_dir)):
        # Determine if this is a train or test character
        if any(char_dir.startswith(prefix) for prefix in train_prefixes):
            dest = 'train'
        elif any(char_dir.startswith(prefix) for prefix in test_prefixes):
            dest = 'test'
        else:
            print(f"Skipping {char_dir} - no matching prefix found")
            continue
        
        # Copy entire character directory to appropriate location
        src_path = os.path.join(src_dir, char_dir)
        dst_path = os.path.join(dst_root, dest, char_dir)
        shutil.copytree(src_path, dst_path)

# Usage
reorganize_dataset(
    src_dir='final_dataset/train',  # Your current single folder dataset
    dst_root='organized_dataset'  # New organized location
)