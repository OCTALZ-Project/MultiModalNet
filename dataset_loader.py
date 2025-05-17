import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
class OCTAMultiModalDataset(Dataset):
    def __init__(self, data_dir, num_classes, octa_transform=None, bscan_transform=None,
                 class_name_map=None):
        """
        Initializes the dataset loader.
        Args:
            data_dir (str): Path to the data directory (e.g., 'DATASET/train', 'DATASET/val').
            num_classes (int): Number of target classes (3 for NORMAL, AMD, DR;
                            4 for NORMAL, AMD, DR, OTHERS).
            octa_transform (callable, optional): Optional transform to be applied on OCTA tensor.
            bscan_transform (callable, optional): Optional transform to be applied on B-scan tensor.
            class_name_map (dict, optional): A dictionary to map directory names on disk
                                            to the desired class names.
                                            E.g., {"AGE_RELATED_MACULAR_DEGENERATION": "AMD"}.
                                            If None or a directory name is not in the map,
                                            the directory name itself (uppercased) is used as the class name.
        """
        self.data_dir = data_dir
        self.octa_transform = octa_transform
        self.bscan_transform = bscan_transform
        self.num_classes = num_classes
        self.class_name_map = class_name_map if class_name_map else {}

        if self.num_classes not in [3, 4]:
            raise ValueError("num_classes must be 3 or 4.")

        # Define expected model classes based on num_classes
        if self.num_classes == 3:
            self.expected_model_classes = ['NORMAL', 'AMD', 'DR']
        else: # self.num_classes == 4
            self.expected_model_classes = ['NORMAL', 'AMD', 'DR', 'OTHERS']

        self.class_to_idx = {name: i for i, name in enumerate(self.expected_model_classes)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        self.ids = []
        self.labels = {} # Stores final mapped label for each subject_id
        self.subject_bscan_files = {}
        self.subject_full_paths = {} # Stores full path to subject directory

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
                print(f"  Skipping directory '{class_dir_on_disk}' (maps to '{target_class_name}') as it's not an expected class for num_classes={self.num_classes}.")
                continue
            
            print(f"  Processing class directory: '{class_dir_on_disk}' as '{target_class_name}'")

            # Iterate through subject_id directories (e.g., 10001, 10002)
            for subject_id_str in os.listdir(class_path):
                subject_path = os.path.join(class_path, subject_id_str)
                if not os.path.isdir(subject_path):
                    continue

                # Check for OCTA files
                octa_file_names_expected = [
                    f'OCTA(FULL)_{subject_id_str}.bmp',
                    f'OCTA(ILM_OPL)_{subject_id_str}.bmp',
                    f'OCTA(OPL_BM)_{subject_id_str}.bmp'
                ]
                octa_files_present = all(os.path.exists(os.path.join(subject_path, fname)) for fname in octa_file_names_expected)

                if not octa_files_present:
                    # print(f"    Warning: Missing one or more OCTA files for ID {subject_id_str} in {subject_path}. Skipping.")
                    continue

                # Find B-scan files (non-OCTA BMP files)
                all_bmp_files_in_subject_dir = [f for f in os.listdir(subject_path) if f.lower().endswith('.bmp')]
                current_bscan_files = []
                for bmp_file in all_bmp_files_in_subject_dir:
                    is_octa = False
                    # Check if the bmp_file matches any OCTA naming patterns
                    for octa_pattern_part in ['OCTA(FULL)', 'OCTA(ILM_OPL)', 'OCTA(OPL_BM)']:
                        if octa_pattern_part in bmp_file and f"_{subject_id_str}.bmp" in bmp_file:
                            is_octa = True
                            break
                    if not is_octa:
                        current_bscan_files.append(bmp_file)
                
                if len(current_bscan_files) != 3:
                    # print(f"    Warning: Found {len(current_bscan_files)} non-OCTA BMP files (expected 3) for ID {subject_id_str} in {subject_path}. Skipping.")
                    continue
                
                try:
                    # Sort B-scan files numerically by their name (e.g., '185.bmp', '200.bmp', '215.bmp')
                    current_bscan_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
                except ValueError:
                    # print(f"    Warning: Could not sort B-scan filenames numerically for ID {subject_id_str} in {subject_path} (files: {current_bscan_files}). Skipping.")
                    continue

                # If all checks pass, add the subject
                self.ids.append(subject_id_str)
                self.labels[subject_id_str] = target_class_name # Store the mapped class name
                self.subject_bscan_files[subject_id_str] = current_bscan_files
                self.subject_full_paths[subject_id_str] = subject_path
                # print(f"      Added subject: {subject_id_str} with label '{target_class_name}'")

        if not self.ids:
            print(f"Warning: No valid samples found in {self.data_dir} matching the criteria for {self.num_classes} classes.")
        else:
            print(f"Dataset loaded from {self.data_dir}.")
            print(f"Total {len(self.ids)} valid samples found and loaded.")
            print(f"Class distribution in this dataset: {Counter(self.labels.values())}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subject_id = self.ids[idx]
        # Retrieve the pre-calculated full path to the subject's directory
        subject_dir = self.subject_full_paths[subject_id]
        
        to_tensor = transforms.ToTensor() # Basic conversion to tensor if no other transform

        # --- Load OCTA images ---
        try:
            octa_full_path = os.path.join(subject_dir, f'OCTA(FULL)_{subject_id}.bmp')
            octa_ilm_opl_path = os.path.join(subject_dir, f'OCTA(ILM_OPL)_{subject_id}.bmp')
            octa_opl_bm_path = os.path.join(subject_dir, f'OCTA(OPL_BM)_{subject_id}.bmp')

            img_octa_full = Image.open(octa_full_path).convert('L')
            img_octa_ilm_opl = Image.open(octa_ilm_opl_path).convert('L')
            img_octa_opl_bm = Image.open(octa_opl_bm_path).convert('L')

            # Convert to tensor before stacking
            t_octa_full = to_tensor(img_octa_full)
            t_octa_ilm_opl = to_tensor(img_octa_ilm_opl)
            t_octa_opl_bm = to_tensor(img_octa_opl_bm)
            
            octa_image_tensor = torch.cat([t_octa_full, t_octa_ilm_opl, t_octa_opl_bm], dim=0)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading OCTA for ID {subject_id} from {subject_dir}: File not found - {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing OCTA for ID {subject_id} from {subject_dir}: {e}")

        # --- Load B-scan images ---
        try:
            bscan_filenames = self.subject_bscan_files[subject_id]
            # Ensure we have exactly 3 B-scan filenames stored
            if len(bscan_filenames) != 3:
                raise RuntimeError(f"Expected 3 B-scan filenames for subject {subject_id}, but found {len(bscan_filenames)}: {bscan_filenames}")

            bscan_1_path = os.path.join(subject_dir, bscan_filenames[0])
            bscan_2_path = os.path.join(subject_dir, bscan_filenames[1])
            bscan_3_path = os.path.join(subject_dir, bscan_filenames[2])

            img_bscan_1 = Image.open(bscan_1_path).convert('L')
            img_bscan_2 = Image.open(bscan_2_path).convert('L')
            img_bscan_3 = Image.open(bscan_3_path).convert('L')

            # Convert to tensor before stacking
            t_bscan_1 = to_tensor(img_bscan_1)
            t_bscan_2 = to_tensor(img_bscan_2)
            t_bscan_3 = to_tensor(img_bscan_3)

            bscan_image_tensor = torch.cat([t_bscan_1, t_bscan_2, t_bscan_3], dim=0)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error loading B-scan for ID {subject_id} from {subject_dir}: File not found - {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing B-scan for ID {subject_id} from {subject_dir}: {e}")

        # Apply transforms if they exist
        if self.octa_transform:
            octa_image_tensor = self.octa_transform(octa_image_tensor)
        if self.bscan_transform:
            bscan_image_tensor = self.bscan_transform(bscan_image_tensor)

        disease_name = self.labels[subject_id] # This is 'NORMAL', 'AMD', 'DR', or 'OTHERS'
        label_idx = self.class_to_idx[disease_name]
        
        return (octa_image_tensor, bscan_image_tensor), label_idx