import os
import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from PIL import Image

class DatasetManager:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def validate_and_parse_dataset(self) -> Dict[str, Any]:
        """
        Validates directory structure, parses metadata CSV if available, and analyzes writer distribution.
        """
        if not os.path.exists(self.dataset_dir):
            return {
                "valid": False,
                "error": f"Dataset directory not found: {self.dataset_dir}"
            }

        samples = []
        csv_path = os.path.join(self.dataset_dir, "metadata.csv")
        
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = os.path.join(self.dataset_dir, row["image_path"]) if not os.path.isabs(row["image_path"]) else row["image_path"]
                    if os.path.exists(img_path):
                        samples.append({
                            "image_path": img_path,
                            "label": int(row["label"]),
                            "writer_id": row.get("writer_id", f"unknown_writer_{len(samples)}"),
                            "source": row.get("source", "custom")
                        })
        else:
            # Fallback to subfolder structure (0/ to 9/)
            for label_digit in range(10):
                folder = os.path.join(self.dataset_dir, str(label_digit))
                if os.path.exists(folder):
                    for fname in os.listdir(folder):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(folder, fname)
                            # Attempt to infer writer_id from filename (e.g., student01_d8.png)
                            writer_id = fname.split('_')[0] if '_' in fname else f"writer_{fname}"
                            samples.append({
                                "image_path": img_path,
                                "label": label_digit,
                                "writer_id": writer_id,
                                "source": "folder"
                            })

        if len(samples) == 0:
            return {
                "valid": False,
                "error": f"No valid digit image samples found in {self.dataset_dir}"
            }

        # Class & Writer distribution
        class_counts = defaultdict(int)
        writer_samples = defaultdict(list)
        for s in samples:
            class_counts[s["label"]] += 1
            writer_samples[s["writer_id"]].append(s)

        return {
            "valid": True,
            "total_samples": len(samples),
            "writer_count": len(writer_samples),
            "class_distribution": dict(class_counts),
            "samples": samples,
            "writer_samples": writer_samples
        }

    def create_writer_independent_splits(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits dataset by writer_id so no writer appears in more than one split.
        """
        data_info = self.validate_and_parse_dataset()
        if not data_info.get("valid"):
            raise ValueError(data_info.get("error", "Invalid dataset"))

        writer_samples = data_info["writer_samples"]
        writers = list(writer_samples.keys())
        random.seed(42)
        random.shuffle(writers)

        n_total = len(writers)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        train_writers = set(writers[:n_train])
        val_writers = set(writers[n_train:n_train + n_val])
        test_writers = set(writers[n_train + n_val:])
        if not test_writers and writers:
            test_writers = val_writers  # Fallback if very few writers exist

        train_samples, val_samples, test_samples = [], [], []

        for w, sample_list in writer_samples.items():
            if w in train_writers:
                train_samples.extend(sample_list)
            elif w in val_writers:
                val_samples.extend(sample_list)
            else:
                test_samples.extend(sample_list)

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "split_summary": {
                "train_writers": len(train_writers),
                "val_writers": len(val_writers),
                "test_writers": len(test_writers),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "test_samples": len(test_samples)
            }
        }
