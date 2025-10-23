import os
import json
import hashlib
from collections import Counter
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def validate_data():
    """
    Loads dataset metadata from CSVs, performs validation + anomaly detection,
    and logs results to MLflow.
    """

    # ==== CONFIG (ปรับได้ตามต้องการ) ====
    EXPECTED_NUM_CLASSES = 30                   # Plants Classification (30 classes)
    VALID_EXTS = {".jpg", ".jpeg", ".png"}      # นามสกุลที่คาดหวัง
    IMAGE_HASH_SAMPLE = 1500                    # จำนวนไฟล์สูงสุดใช้ทำ hashing หา duplicate content
    CHECK_IMAGE_CONTENT = True                  # เปิดปิดการลองเปิดภาพเพื่อตรวจสอบความเสียหาย
    MIN_BYTES = 256                             # ขนาดไฟล์อย่างน้อย (bytes) เพื่อกันไฟล์เสีย/ว่าง
    RANDOM_STATE = 42
    # =====================================

    # Set the experiment name for this step
    mlflow.set_experiment("Data Validation")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data validation run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_validation")

        validation_data_dir = "validation_data"
        os.makedirs(validation_data_dir, exist_ok=True)

        def data_validation(data: Dict[str, str]) -> Dict:
            all_df = []
            for section, path in data.items():
                if section == "root_dir":
                    continue
                df = pd.read_csv(path)
                # shuffle เพื่อสุ่ม sampling ให้เสถียร
                df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
                df["sections"] = section
                all_df.append(df)

            merged_df = pd.concat(all_df, ignore_index=True)

            # สร้างคอลัมน์จากโครงสร้างไดเรกทอรี
            # เดิม dataset ใช้คอลัมน์ "image:FILE"
            merged_df["category"] = merged_df["image:FILE"].apply(
                lambda x: os.path.split(x)[0].split("/")[-1]
            )
            merged_df.rename(columns={"image:FILE": "file_path", "category": "class"}, inplace=True)

            # ต่อ root_dir + ปรับเป็น path ปกติ
            merged_df["file_path"] = merged_df["file_path"].apply(
                lambda x: os.path.join(data["root_dir"], os.path.join(*x.split("/")))
            )
            merged_df["file_path"] = merged_df["file_path"].apply(lambda x: x.replace("\\", "/"))

            class_names = merged_df["class"].unique()

            number_per_class = {}
            for section in data.keys():
                if section == "root_dir":
                    continue
                section_df = merged_df[merged_df["sections"] == section].copy()
                number_per_class[section] = int(section_df["class"].nunique())
                section_df.drop(columns=["sections"], inplace=True)
                section_df.reset_index(drop=True, inplace=True)
                section_df.to_csv(os.path.join(validation_data_dir, f"{section}.csv"), index=False)

            return {
                "merged_df": merged_df,
                "class_names": class_names,
                "Number of class": len(class_names),
                "Number of data": len(merged_df),
                "number_per_class": number_per_class,
            }

        # เลือกชุดข้อมูล
        if os.path.exists("Dataset"):
            data_path = {
                "root_dir": r"Dataset",
                "train": "Dataset/train.csv",
                "val": "Dataset/val.csv",
                "test": "Dataset/test.csv",
            }
        else:
            data_path = {
                "root_dir": r"Dataset_github",
                "train": "Dataset_github/train.csv",
                "val": "Dataset_github/val.csv",
                "test": "Dataset_github/test.csv",
            }

        meta_data = data_validation(data_path)
        merged_df = meta_data["merged_df"]
        class_names = meta_data["class_names"]
        number_of_class = meta_data["Number of class"]
        number_of_data = meta_data["Number of data"]
        number_per_class = meta_data["number_per_class"]

        # ---------- (A) LOG BASELINE VALIDATION ----------
        mlflow.log_metric("num_rows", number_of_data)
        mlflow.log_metric("num_classes_metric", number_of_class)   # เปลี่ยนชื่อ metric ให้ชัด
        mlflow.log_param("num_classes", number_of_class)
        mlflow.log_param("class_names", json.dumps(sorted(class_names.tolist())))
        mlflow.log_param("num_train_classes", number_per_class.get("train", 0))
        mlflow.log_param("num_val_classes", number_per_class.get("val", 0))
        mlflow.log_param("num_test_classes", number_per_class.get("test", 0))

        # ---------- (B) ANOMALY DETECTION ----------
        anomalies = {
            "missing_files": [],
            "bad_extensions": [],
            "zero_byte_files": [],
            "unreadable_images": [],
            "non_positive_dimensions": [],
            "duplicate_paths": [],
            "duplicate_hashes": [],
            "class_count_outliers": [],
            "class_count_missing_or_extra": {},
        }

        # (B1) ตรวจ class coverage
        anomalies["class_count_missing_or_extra"] = {
            "expected_num_classes": EXPECTED_NUM_CLASSES,
            "observed_num_classes": int(number_of_class),
            "status": (
                "match" if number_of_class == EXPECTED_NUM_CLASSES
                else ("less" if number_of_class < EXPECTED_NUM_CLASSES else "more")
            ),
        }

        # (B2) ตรวจความไม่สมดุลของจำนวนภาพต่อคลาส
        counts = merged_df.groupby("class")["file_path"].count().sort_values()
        counts_df = counts.reset_index().rename(columns={"file_path": "count"})
        counts_df["zscore"] = (counts_df["count"] - counts_df["count"].mean()) / (counts_df["count"].std(ddof=1) + 1e-9)

        # ใช้เกณฑ์ |z| >= 2 เป็น outlier แบบคร่าว ๆ
        outlier_rows = counts_df[np.abs(counts_df["zscore"]) >= 2.0].copy()
        if not outlier_rows.empty:
            anomalies["class_count_outliers"] = outlier_rows.to_dict(orient="records")

        counts_df.to_csv(os.path.join(validation_data_dir, "class_counts.csv"), index=False)

        # (B3) ตรวจเส้นทางไฟล์ + นามสกุล + ไฟล์ว่าง
        def file_ext(path: str) -> str:
            return os.path.splitext(path)[1].lower()

        # หา duplicate path
        path_counts = Counter(merged_df["file_path"].tolist())
        duplicate_path_list = [p for p, c in path_counts.items() if c > 1]
        anomalies["duplicate_paths"] = duplicate_path_list

        # ตรวจการมีจริงของไฟล์ + นามสกุล + ไฟล์ว่าง
        for p in merged_df["file_path"]:
            if not os.path.exists(p):
                anomalies["missing_files"].append(p)
                continue
            ext = file_ext(p)
            if ext not in VALID_EXTS:
                anomalies["bad_extensions"].append(p)
            try:
                size = os.path.getsize(p)
                if size < MIN_BYTES:
                    anomalies["zero_byte_files"].append(p)
            except OSError:
                anomalies["zero_byte_files"].append(p)

        # (B4) ตรวจเปิดภาพ/มิติภาพ (เลือกได้)
        if CHECK_IMAGE_CONTENT:
            for p in merged_df["file_path"]:
                if not os.path.exists(p):
                    continue
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        if w <= 0 or h <= 0:
                            anomalies["non_positive_dimensions"].append(p)
                        # ทดสอบโหลดเล็กน้อย
                        im.verify()  # ตรวจความถูกต้องของไฟล์
                except (UnidentifiedImageError, OSError):
                    anomalies["unreadable_images"].append(p)

        # (B5) ตรวจ duplicate content ด้วย hashing (สุ่มบางส่วนเพื่อประหยัดเวลา)
        # เลือก sample ไม่เกิน IMAGE_HASH_SAMPLE
        sample_paths = merged_df["file_path"].sample(
            n=min(IMAGE_HASH_SAMPLE, len(merged_df)),
            random_state=RANDOM_STATE
        ).tolist()

        hash_map = {}
        for p in sample_paths:
            if not os.path.exists(p):
                continue
            try:
                # MD5 ของเนื้อไฟล์
                md5 = hashlib.md5()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        md5.update(chunk)
                h = md5.hexdigest()
                hash_map.setdefault(h, []).append(p)
            except OSError:
                pass

        duplicate_hash_groups = [v for v in hash_map.values() if len(v) > 1]
        anomalies["duplicate_hashes"] = duplicate_hash_groups

        # ---------- (C) สรุป & LOG ----------
        # สร้าง DataFrame รายการ anomaly ทีละประเภทเป็นไฟล์
        def _write_list(name: str, data: List):
            path = os.path.join(validation_data_dir, f"{name}.csv")
            if isinstance(data, list) and data and isinstance(data[0], list):
                # กรณีกลุ่มไฟล์ (เช่น duplicate_hashes)
                flat_rows = []
                for grp in data:
                    flat_rows.append({"group_size": len(grp), "files": ";".join(grp)})
                pd.DataFrame(flat_rows).to_csv(path, index=False)
            else:
                pd.DataFrame({"path": data}).to_csv(path, index=False)

        _write_list("missing_files", anomalies["missing_files"])
        _write_list("bad_extensions", anomalies["bad_extensions"])
        _write_list("zero_byte_files", anomalies["zero_byte_files"])
        _write_list("unreadable_images", anomalies["unreadable_images"])
        _write_list("non_positive_dimensions", anomalies["non_positive_dimensions"])
        _write_list("duplicate_paths", anomalies["duplicate_paths"])
        _write_list("duplicate_hashes", anomalies["duplicate_hashes"])

        outlier_path = os.path.join(validation_data_dir, "class_count_outliers.csv")
        pd.DataFrame(anomalies["class_count_outliers"]).to_csv(outlier_path, index=False)

        with open(os.path.join(validation_data_dir, "anomalies_summary.json"), "w", encoding="utf-8") as f:
            json.dump(anomalies, f, ensure_ascii=False, indent=2)

        # Metrics/params สำหรับ anomaly
        mlflow.log_param("expected_num_classes", EXPECTED_NUM_CLASSES)
        mlflow.log_metric("classes_observed", number_of_class)
        mlflow.log_metric("missing_files_count", len(anomalies["missing_files"]))
        mlflow.log_metric("bad_ext_count", len(anomalies["bad_extensions"]))
        mlflow.log_metric("zero_byte_count", len(anomalies["zero_byte_files"]))
        mlflow.log_metric("unreadable_img_count", len(anomalies["unreadable_images"]))
        mlflow.log_metric("non_pos_dim_count", len(anomalies["non_positive_dimensions"]))
        mlflow.log_metric("dup_path_count", len(anomalies["duplicate_paths"]))
        mlflow.log_metric("dup_hash_groups", len(anomalies["duplicate_hashes"]))
        mlflow.log_metric("class_outlier_count", len(anomalies["class_count_outliers"]))

        # สถานะรวม (ผ่านเมื่อข้อมูลมีอย่างน้อย 1 แถว, คลาส ≥1 และไม่มี critical anomalies: missing_files/unreadable_images มากผิดปกติ)
        validation_status = "passed" if (number_of_data > 0 and number_of_class > 0) else "failed"
        if (len(anomalies["missing_files"]) > 0 or len(anomalies["unreadable_images"]) > 0):
            validation_status = "warn" if validation_status == "passed" else validation_status

        mlflow.set_tag("anomaly.status", validation_status)
        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # Log artifacts
        mlflow.log_artifacts(validation_data_dir, artifact_path="validation_data")
        print("Logged validation data & anomaly reports as artifacts in MLflow.")

        print("-" * 50)
        print("Data validation run finished. Use this Run ID for the next step:")
        print(f"Validation Run ID: {run_id}")
        print("-" * 50)

        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"validation_run_id={run_id}", file=f)
        else:
            with open("run_id.json", "w") as f:
                json.dump({"validation_run_id": run_id}, f)


if __name__ == "__main__":
    validate_data()
