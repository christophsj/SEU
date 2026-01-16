#!/usr/bin/env python3
"""
Generate translated entity names from DBpedia entity URIs
This script extracts entity names from URIs and creates simple word tokenizations
"""

import json
import os
import re
from urllib.parse import unquote


def extract_entity_name_from_uri(uri):
    """
    Extract entity name from DBpedia URI
    Example: http://dbpedia.org/resource/Barack_Obama -> ["barack", "obama"]
    """
    # Get the last part of the URI (after last /)
    entity_part = uri.split("/")[-1]

    # URL decode (e.g., %20 -> space)
    entity_part = unquote(entity_part)

    # Replace underscores and other separators with spaces
    entity_part = entity_part.replace("_", " ")
    entity_part = entity_part.replace("-", " ")

    # Remove parentheses content and other special characters
    entity_part = re.sub(r"\([^)]*\)", "", entity_part)
    entity_part = re.sub(r"[,;:]", " ", entity_part)

    # Split into words and convert to lowercase
    words = entity_part.lower().split()

    # Remove empty strings and very short words
    words = [w.strip() for w in words if len(w.strip()) > 0]

    return words


def generate_entity_names_for_dataset(kg_folder, output_filename):
    """
    Generate entity names JSON file for a dataset
    """
    print(f"\n{'='*80}")
    print(f"Processing: {kg_folder}")
    print(f"{'='*80}")

    entity_names = []

    # Read both entity ID files
    ent_files = ["ent_ids_1", "ent_ids_2"]

    for ent_file in ent_files:
        filepath = os.path.join("KGs", kg_folder, ent_file)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        print(f"Reading {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                entity_id = int(parts[0])
                entity_uri = parts[1]

                # Extract words from URI
                words = extract_entity_name_from_uri(entity_uri)

                # Add to entity names list
                entity_names.append([entity_id, words])

    # Sort by entity ID
    entity_names.sort(key=lambda x: x[0])

    print(f"Extracted {len(entity_names)} entity names")

    # Save to JSON file
    output_path = os.path.join("translated_ent_name", output_filename)
    os.makedirs("translated_ent_name", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entity_names, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")
    print(f"Sample entries:")
    for i in range(min(5, len(entity_names))):
        print(f"  {entity_names[i]}")


def main():
    """
    Generate entity names for all datasets
    """
    # Mapping of KG folders to output filenames
    datasets = {
        "fr_en": "dbp_fr_en.json",
        "ja_en": "dbp_ja_en.json",
        "zh_en": "dbp_zh_en.json",
        "EN_DE_15K_V2": "en_de_15k_v2.json",
        "EN_FR_15K_V2": "en_fr_15k_v2.json",
        "D_W_15K_V2": "d_w_15k_v2.json",
        "D_Y_15K_V2": "d_y_15k_v2.json",
    }

    print("=" * 80)
    print("Entity Name Generator")
    print("=" * 80)
    print(f"Will generate translated entity names for {len(datasets)} datasets")
    print()

    for kg_folder, output_filename in datasets.items():
        kg_path = os.path.join("KGs", kg_folder)

        if not os.path.exists(kg_path):
            print(f"Skipping {kg_folder} - directory not found")
            continue

        try:
            generate_entity_names_for_dataset(kg_folder, output_filename)
        except Exception as e:
            print(f"ERROR processing {kg_folder}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
