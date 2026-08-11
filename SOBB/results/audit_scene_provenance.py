import argparse
import hashlib
import json
import os
import sys


def compute_sha256(filepath):
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_images(directory, extensions=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    """Find all image files in a directory tree."""
    images = []
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                images.append(os.path.join(root, fname))
    return images


def extract_chip_info(filepath, dataset_name):
    """Extract chip coordinate information from filename and image metadata."""
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0]
    parent_scene_id = None
    chip_coords = None
    if dataset_name == "SSDD":
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            parent_scene_id = "SSDD_scene_{}".format(parts[0])
    elif dataset_name == "RSDD":
        parts = stem.split("_")
        if len(parts) >= 2:
            parent_scene_id = "RSDD_{}".format(parts[0])
    return {
        "original_filename": basename,
        "parent_scene_id": parent_scene_id,
        "chip_coords": chip_coords,
    }


def audit_dataset(dataset_dir, dataset_name, template_scenes):
    """Audit a single dataset and return per-image provenance records."""
    if not os.path.isdir(dataset_dir):
        print("Warning: {} directory not found: {}".format(dataset_name, dataset_dir))
        return {}
    images = find_images(dataset_dir)
    if not images:
        print("Warning: no image files found in {}".format(dataset_dir))
        return {}
    print("Auditing {}: {} images".format(dataset_name, len(images)))
    records = {}
    for img_path in images:
        sha = compute_sha256(img_path)
        info = extract_chip_info(img_path, dataset_name)
        info["sha256"] = sha
        info["full_path"] = img_path
        records[info["original_filename"]] = info
    return records


def check_overlap(ssdd_records, rsdd_records):
    """Check for potential inter-split data leakage."""
    issues = []
    all_hashes = {}
    for ds_name, records in [("SSDD", ssdd_records), ("RSDD", rsdd_records)]:
        for fname, info in records.items():
            sha = info["sha256"]
            if sha in all_hashes:
                other_ds, other_fname = all_hashes[sha]
                issues.append("DUPLICATE: {}:{} and {}:{} have same SHA-256".format(
                    other_ds, other_fname, ds_name, fname))
            else:
                all_hashes[sha] = (ds_name, fname)
    if not issues:
        print("No SHA-256 duplicates found across datasets.")
    else:
        for issue in issues:
            print("WARNING: " + issue)
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Audit SSDD and RSDD scene provenance with SHA-256 checksums.")
    parser.add_argument("--ssdd-dir", default=None,
                        help="Path to the SSDD dataset directory")
    parser.add_argument("--rsdd-dir", default=None,
                        help="Path to the RSDD dataset directory")
    parser.add_argument("--output",
                        default="results/scene_sensor_slice_map_audited.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    if not args.ssdd_dir and not args.rsdd_dir:
        parser.error("At least one of --ssdd-dir or --rsdd-dir is required.")

    template_path = os.path.join("results", "scene_sensor_slice_map.json")
    template = {}
    if os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

    ssdd_records = {}
    rsdd_records = {}
    if args.ssdd_dir:
        ssdd_records = audit_dataset(args.ssdd_dir, "SSDD",
                                     template.get("datasets", {})
                                     .get("SSDD", {}).get("scenes", {}))
    if args.rsdd_dir:
        rsdd_records = audit_dataset(args.rsdd_dir, "RSDD",
                                     template.get("datasets", {})
                                     .get("RSDD", {}).get("scenes", {}))

    check_overlap(ssdd_records, rsdd_records)

    output = {
        "description": "Audited scene provenance map with SHA-256 checksums.",
        "ssdd_images": ssdd_records,
        "rsdd_images": rsdd_records,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nAudit complete. Output written to {}".format(args.output))
    print("  SSDD images audited: {}".format(len(ssdd_records)))
    print("  RSDD images audited: {}".format(len(rsdd_records)))


if __name__ == "__main__":
    main()