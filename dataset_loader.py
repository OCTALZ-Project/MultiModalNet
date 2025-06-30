import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

class OCTAMultiModalDataset(Dataset):
    def __init__(self, data_dir, num_classes, projection_transform=None, bscan_transform=None,
                 class_name_map=None, verbose=False, input_size=224):
        """
        Initializes the dataset loader.
        Args:
            data_dir (str): Path to the data directory (e.g., 'DATASET/train', 'DATASET/val').
            num_classes (int): Number of target classes. The dataset will attempt to find this many
                               classes in the directory structure. If the number doesn't match,
                               a warning will be displayed.
            projection_transform (callable, optional): Optional transform to be applied on projection tensor.
            bscan_transform (callable, optional): Optional transform to be applied on B-scan tensor.
            class_name_map (dict, optional): A dictionary to map directory names on disk
                                            to the desired class names.
                                            E.g., {"AGE_RELATED_MACULAR_DEGENERATION": "AMD"}.
                                            If None or a directory name is not in the map,
                                            the directory name itself (uppercased) is used as the class name.
            verbose (bool, optional): If True, print detailed information during dataset loading.
            input_size (int, optional): Desired image size for resizing (default: 224)
        """
        self.data_dir = data_dir
        self.projection_transform = projection_transform
        self.bscan_transform = bscan_transform
        self.num_classes = num_classes
        self.class_name_map = class_name_map if class_name_map else {}
        self.verbose = verbose
        self.input_size = input_size
        # Supported image extensions
        self.supported_extensions = ['.bmp', '.png', '.jpg', '.jpeg']

        # Dynamically discover classes from directory structure
        self.expected_model_classes = []
        
        # First scan through directories to collect available classes
        for class_dir_on_disk in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir_on_disk)
            if os.path.isdir(class_path):
                target_class_name = self.class_name_map.get(class_dir_on_disk, class_dir_on_disk.upper())
                if target_class_name not in self.expected_model_classes:
                    self.expected_model_classes.append(target_class_name)
        
        # If we couldn't find enough directories or found too many
        if len(self.expected_model_classes) != self.num_classes:
            print(f"Warning: Found {len(self.expected_model_classes)} class directories but num_classes is {self.num_classes}.")
            print(f"Using discovered classes: {self.expected_model_classes}")
            
        # Sort classes to ensure consistent ordering
        self.expected_model_classes.sort()

        self.class_to_idx = {name: i for i, name in enumerate(self.expected_model_classes)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        self.ids = []
        self.labels = {} # Stores final mapped label for each subject_id
        self.subject_bscan_files = {}
        self.subject_full_paths = {} # Stores full path to subject directory
        self.projection_files = {} # Stores paths to projection files for each subject

        if self.verbose:
            print(f"Loading data from directory: {self.data_dir} for {self.num_classes} classes.")
            print(f"Expecting class directories corresponding to: {self.expected_model_classes}")
            if self.class_name_map:
                print(f"Using class name map: {self.class_name_map}")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Iterate through class directories (e.g., NORMAL, AMD, DR)
        for class_dir_on_disk in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir_on_disk)
            if not os.path.isdir(class_path):
                continue

            # Map directory name to target class name
            # If "OTHERS_DIR" is on disk and class_name_map is {"OTHERS_DIR": "OTHERS"},
            # then target_class_name becomes "OTHERS".
            # If "AMD" is on disk and no map, target_class_name becomes "AMD".
            target_class_name = self.class_name_map.get(class_dir_on_disk, class_dir_on_disk.upper())

            if target_class_name not in self.expected_model_classes:
                if self.verbose:
                    print(f"  Skipping directory '{class_dir_on_disk}' (maps to '{target_class_name}') as it's not an expected class for num_classes={self.num_classes}.")
                continue
            
            if self.verbose:
                print(f"  Processing class directory: '{class_dir_on_disk}' as '{target_class_name}'")

            # Iterate through subject_id directories (e.g., 10001, 10002)
            for subject_id_str in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject_id_str)
                if not os.path.isdir(subject_path):
                    continue

                # New directory structure has projection and bscan subdirectories
                projection_dir = os.path.join(subject_path, "projection")
                bscan_dir = os.path.join(subject_path, "bscan")
                
                # Skip if either directory doesn't exist
                if not (os.path.isdir(projection_dir) and os.path.isdir(bscan_dir)):
                    if self.verbose:
                        print(f"    Warning: Missing projection or bscan directory for ID {subject_id_str} in {subject_path}. Skipping.")
                    continue
                
                # Check for OCTA files with any supported extension
                projection_files = self._find_projection_files(projection_dir, subject_id_str)
                if not projection_files or len(projection_files) != 3:
                    if self.verbose:
                        print(f"    Warning: Could not find suitable projection files for ID {subject_id_str} in {projection_dir}. Skipping.")
                    continue

                # Find B-scan files in the bscan directory
                bscan_files = self._find_bscan_files(bscan_dir)
                if len(bscan_files) != 3:
                    if self.verbose:
                        print(f"    Warning: Found {len(bscan_files)} B-scan files (expected 3) for ID {subject_id_str} in {bscan_dir}. Skipping.")
                    continue
                
                try:
                    # Sort B-scan files numerically by their name (e.g., '185.bmp', '200.bmp', '215.bmp')
                    bscan_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                except ValueError:
                    if self.verbose:
                        print(f"    Warning: Could not sort B-scan filenames numerically for ID {subject_id_str}. Skipping.")
                    continue

                # If all checks pass, add the subject
                self.ids.append(subject_id_str)
                self.labels[subject_id_str] = target_class_name # Store the mapped class name
                self.subject_bscan_files[subject_id_str] = [os.path.basename(f) for f in bscan_files]
                self.subject_full_paths[subject_id_str] = subject_path
                self.projection_files[subject_id_str] = projection_files
                if self.verbose:
                    # Get the types of projection files (specific OCTA or generic)
                    projection_types = []
                    for path in projection_files:
                        filename = os.path.basename(path)
                        if "(FULL)" in filename:
                            projection_types.append("FULL")
                        elif "(ILM_OPL)" in filename:
                            projection_types.append("ILM_OPL")
                        elif "(OPL_BM)" in filename:
                            projection_types.append("OPL_BM")
                        else:
                            projection_types.append("Generic")
                    if len(set(projection_types)) == 1 and projection_types[0] == "Generic":
                        print(f"      Added subject: {subject_id_str} with label '{target_class_name}' (using generic projection)")
                    else:
                        print(f"      Added subject: {subject_id_str} with label '{target_class_name}' (using specific projections)")

        if not self.ids:
            print(f"Warning: No valid samples found in {self.data_dir} matching the criteria for {self.num_classes} classes.")
        else:
            print(f"Dataset loaded from {self.data_dir}.")
            print(f"Total {len(self.ids)} valid samples found and loaded.")
            print(f"Class distribution in this dataset: {Counter(self.labels.values())}")

    def _find_projection_files(self, projection_dir, subject_id_str):
        """Find projection files with any supported extension in the projection directory.
        
        If specific projection files (FULL, ILM_OPL, OPL_BM) are found, they are returned.
        If only a single generic projection file is found, it is used for all three channels.
        """
        projection_keys = {'FULL': None, 'ILM_OPL': None, 'OPL_BM': None}
        
        # List all files in the projection directory
        files_in_dir = [f for f in os.listdir(projection_dir) 
                      if os.path.isfile(os.path.join(projection_dir, f)) and
                      any(f.lower().endswith(ext) for ext in self.supported_extensions)]
        
        # First, try to find the specific OCTA files
        for filename in files_in_dir:
            file_path = os.path.join(projection_dir, filename)
                
            # Check if the file matches any OCTA pattern
            for key in projection_keys.keys():
                if f"({key})" in filename and subject_id_str in filename:
                    projection_keys[key] = file_path
                    break
        
        # If all three specific OCTA files were found, return them
        if all(projection_keys.values()):
            return list(projection_keys.values())
        
        # If not all specific OCTA files were found, look for a single generic projection file
        # Common names for generic projection files
        generic_names = ['projection', 'oct', 'scan', 'image']
        
        # Reset and look for a generic projection file
        generic_projection = None
        for filename in files_in_dir:
            file_base = os.path.splitext(filename.lower())[0]
            if any(name in file_base for name in generic_names):
                generic_projection = os.path.join(projection_dir, filename)
                break
        
        # If no named file is found, just take the first image file
        if not generic_projection and files_in_dir:
            generic_projection = os.path.join(projection_dir, files_in_dir[0])
        
        # If a generic projection file is found, duplicate it for all three channels
        if generic_projection:
            return [generic_projection, generic_projection, generic_projection]
            
        # If no suitable files were found, return an empty list
        return []

    def _find_bscan_files(self, bscan_dir):
        """Find B-scan files with any supported extension in the bscan directory."""
        bscan_files = []
        
        # List all files in the bscan directory
        for filename in os.listdir(bscan_dir):
            file_path = os.path.join(bscan_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            # Check if the file has a supported extension
            if any(filename.lower().endswith(ext) for ext in self.supported_extensions):
                try:
                    # Try to parse the filename as a number (typical for B-scan like 185.bmp)
                    int(os.path.splitext(filename)[0])
                    bscan_files.append(file_path)
                except ValueError:
                    # If filename can't be converted to int, it might not be a B-scan file
                    pass
                    
        return bscan_files

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subject_id = self.ids[idx]
        # Retrieve the pre-calculated full path to the subject's directory
        subject_dir = self.subject_full_paths[subject_id]
        
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((self.input_size, self.input_size))
        # --- Load projection images from saved paths ---
        try:
            projection_paths = self.projection_files[subject_id]
            if len(projection_paths) != 3:
                raise RuntimeError(f"Expected 3 projection paths for subject {subject_id}, but found {len(projection_paths)}")
            using_generic_projection = projection_paths[0] == projection_paths[1] == projection_paths[2]
            if using_generic_projection and self.verbose:
                print(f"Loading generic projection file for subject {subject_id}: {os.path.basename(projection_paths[0])}")
            img_projection_full = Image.open(projection_paths[0]).convert('L')
            img_projection_ilm_opl = Image.open(projection_paths[1]).convert('L')
            img_projection_opl_bm = Image.open(projection_paths[2]).convert('L')
            # Resize before tensor
            img_projection_full = resize(img_projection_full)
            img_projection_ilm_opl = resize(img_projection_ilm_opl)
            img_projection_opl_bm = resize(img_projection_opl_bm)
            t_projection_full = to_tensor(img_projection_full)
            t_projection_ilm_opl = to_tensor(img_projection_ilm_opl)
            t_projection_opl_bm = to_tensor(img_projection_opl_bm)
            projection_image_tensor = torch.cat([t_projection_full, t_projection_ilm_opl, t_projection_opl_bm], dim=0)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading projection for ID {subject_id} from {subject_dir}: File not found - {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing projection for ID {subject_id} from {subject_dir}: {e}")

        # --- Load B-scan images from the bscan subdirectory ---
        try:
            bscan_filenames = self.subject_bscan_files[subject_id]
            bscan_dir = os.path.join(subject_dir, "bscan")
            if len(bscan_filenames) != 3:
                raise RuntimeError(f"Expected 3 B-scan filenames for subject {subject_id}, but found {len(bscan_filenames)}: {bscan_filenames}")
            bscan_1_path = os.path.join(bscan_dir, bscan_filenames[0])
            bscan_2_path = os.path.join(bscan_dir, bscan_filenames[1])
            bscan_3_path = os.path.join(bscan_dir, bscan_filenames[2])
            img_bscan_1 = Image.open(bscan_1_path).convert('L')
            img_bscan_2 = Image.open(bscan_2_path).convert('L')
            img_bscan_3 = Image.open(bscan_3_path).convert('L')
            # Resize before tensor
            img_bscan_1 = resize(img_bscan_1)
            img_bscan_2 = resize(img_bscan_2)
            img_bscan_3 = resize(img_bscan_3)
            t_bscan_1 = to_tensor(img_bscan_1)
            t_bscan_2 = to_tensor(img_bscan_2)
            t_bscan_3 = to_tensor(img_bscan_3)
            bscan_image_tensor = torch.cat([t_bscan_1, t_bscan_2, t_bscan_3], dim=0)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading B-scan for ID {subject_id} from {bscan_dir}: File not found - {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing B-scan for ID {subject_id} from {bscan_dir}: {e}")

        # Apply transforms if they exist
        if self.projection_transform:
            projection_image_tensor = self.projection_transform(projection_image_tensor)
        if self.bscan_transform:
            bscan_image_tensor = self.bscan_transform(bscan_image_tensor)

        class_name = self.labels[subject_id] # The class name for this subject
        label_idx = self.class_to_idx[class_name]
        
        return (projection_image_tensor, bscan_image_tensor), label_idx