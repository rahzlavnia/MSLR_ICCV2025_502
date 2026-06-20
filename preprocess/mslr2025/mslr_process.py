import os
import json
from tqdm import tqdm


# =========================================================
# PROJECT PATH
# =========================================================

def get_project_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

# Project root directory
PROJECT_ROOT = get_project_root()

# Preprocess root directory
PREPROCESS_ROOT = os.path.dirname(__file__)

# Dataset root directory
DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "mslr2025"
)

# Create dataset root directory
os.makedirs(DATASET_ROOT, exist_ok=True)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

# Dataset splits
DATASET_SPLITS = {
    "train": {
        "folder": "SI",
        "file": "train_list.txt",
    },

    "dev": {
        "folder": "SI",
        "file": "dev_list.txt",
    },

    "test_si_major": {
        "folder": "SI-MAJ",
        "file": "test_list.txt",
    },

    "test_si_minor": {
        "folder": "SI-MIN",
        "file": "test_list.txt",
    },
}


# =========================================================
# CORE FUNCTIONS
# =========================================================

def sign_dict_update(total_dict, info):
    for item in info:
        split_label = item['gloss_sequence'].split()
        for gloss in split_label:
            total_dict[gloss] = (
                total_dict.get(gloss, 0) + 1
            )
    return total_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for item in info:
            f.write(
                f"{item['video_id']} "
                f"1 "
                f"{item['signer']} "
                f"0.0 "
                f"1.79769e+308 "
                f"{item['gloss_sequence']}\n"
            )


def info2dict(anno_path):
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found:\n{anno_path}"
        )
    with open(anno_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if (
        len(lines) > 0 and
        (
            "video" in lines[0].lower() or
            "gloss" in lines[0].lower()
        )
    ):
        lines = lines[1:]
    info_list = []
    for line in tqdm(lines):
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue
        video_id = parts[0]
        gloss_seq = parts[1]
        split_vid = video_id.split('_')
        signer = split_vid[0]
        sentence_id = split_vid[1]
        info_list.append({
            "signer": signer,
            "video_id": video_id,
            "gloss_sequence": gloss_seq.strip(),
            "sentence_id": sentence_id,
            "original_info": line,
        })
    return info_list


def save_json(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":    
    print("\n===================================")
    print("MSLR2025 PREPROCESSING START")
    print("===================================\n")

    # =====================================================
    # GLOBAL GLOSS DICTIONARY
    # =====================================================
    global_sign_dict = dict()

    # =====================================================
    # PROCESS EACH SPLIT
    # =====================================================
    for split_name, cfg in DATASET_SPLITS.items():
        folder = cfg["folder"]
        file_name = cfg["file"]
        anno_path = os.path.join(
            PREPROCESS_ROOT,
            folder,
            file_name
        )
        print(f"\nProcessing: {split_name}")
        # =================================================
        # LOAD SPLIT
        # =================================================
        split_info = info2dict(anno_path)
        # =================================================
        # SAVE METADATA JSON
        # =================================================
        json_save_path = os.path.join(
            DATASET_ROOT,
            f"{split_name}_info.json"
        )
        save_json(split_info, json_save_path)
        # =================================================
        # SAVE STM GROUND TRUTH
        # =================================================
        stm_save_path = os.path.join(
            DATASET_ROOT,
            f"mslr-groundtruth-{split_name}.stm"
        )
        generate_gt_stm(
            split_info,
            stm_save_path
        )
        # =================================================
        # BUILD GLOBAL VOCABULARY
        # ONLY FROM TRAIN + DEV
        # =================================================
        if split_name in ["train", "dev"]:
            sign_dict_update(
                global_sign_dict,
                split_info
            )
    # =====================================================
    # SORT GLOSS DICTIONARY
    # =====================================================
    global_sign_dict = sorted(
        global_sign_dict.items(),
        key=lambda d: d[0]
    )
    # =====================================================
    # BUILD GLOSS MAPPING
    # =====================================================
    save_dict = {
        "id2gloss": {},
        "gloss2id": {},
    }
    for idx, (gloss, freq) in enumerate(global_sign_dict):
        gloss_index = idx + 1
        save_dict["gloss2id"][gloss] = {
            "index": gloss_index,
            "frequency": freq,
        }
        save_dict["id2gloss"][gloss_index] = {
            "gloss": gloss,
            "frequency": freq,
        }
    # =====================================================
    # SAVE GLOBAL GLOSS DICTIONARY
    # =====================================================
    gloss_dict_path = os.path.join(
        DATASET_ROOT,
        "global_gloss_dict.json"
    )
    save_json(save_dict, gloss_dict_path)
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n===================================")
    print("PREPROCESSING FINISHED")
    print("===================================\n")