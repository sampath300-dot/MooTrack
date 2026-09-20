import collections
import numpy as np
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "oliveirabruno01/openfarm-ungulate-valence"


def main():
    print("=" * 70)
    print("MOOTRACK - CATTLE DATASET DETAILED ANALYSIS")
    print("=" * 70)
    
    dataset = load_dataset(DATASET_NAME)
    
    print("\n--- Available Splits on Hugging Face Hub ---")
    for split_name, split in dataset.items():
        print(f"  • {split_name}: {len(split)} rows")
        
    # Aggregate raw dataset (train_raw + test_raw)
    train_raw = dataset["train_raw"].to_pandas()
    test_raw = dataset["test_raw"].to_pandas()
    all_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    
    print(f"\nTotal call-level records across all ungulates: {len(all_raw)}")
    
    # Filter for Cow/cattle
    cow_df = all_raw[all_raw["species"] == "Cow/cattle"].copy()
    print(f"\n--- Cattle / Cow Dataset Overview ---")
    print(f"Total Cattle calls: {len(cow_df)}")
    
    print("\n1. Valence Distribution in Cattle:")
    valence_counts = cow_df["valence"].value_counts()
    for val, count in valence_counts.items():
        print(f"   - {val}: {count} ({count / len(cow_df) * 100:.2f}%)")
        
    print("\n2. Grouping & Leakage Identifiers:")
    print(f"   - Unique animal_id: {cow_df['animal_id'].nunique()}")
    print(f"   - Unique source_group_id: {cow_df['source_group_id'].nunique()}")
    print(f"   - Unique audio_sha256 (hash): {cow_df['audio_sha256'].nunique()} (Total calls: {len(cow_df)})")
    
    print("\n3. Audio Signal Properties:")
    print(f"   - Sample rates (Hz): {cow_df['sample_rate_hz'].unique().tolist()}")
    print(f"   - Channels: {cow_df['channels'].unique().tolist()}")
    print(f"   - Min duration:  {cow_df['duration_sec'].min():.3f}s")
    print(f"   - Max duration:  {cow_df['duration_sec'].max():.3f}s")
    print(f"   - Mean duration: {cow_df['duration_sec'].mean():.3f}s")
    print(f"   - Median duration: {cow_df['duration_sec'].median():.3f}s")

    print("\n4. Context Breakdown:")
    context_ct = pd.crosstab(cow_df["context"], cow_df["valence"], margins=True)
    print(context_ct)
    
    print("\n5. License & Attribution:")
    print(f"   - License: {cow_df['license'].iloc[0]}")
    print(f"   - Source reference: {cow_df['source_reference'].iloc[0]}")
    print(f"   - Source DOI: {cow_df['source_doi'].iloc[0]}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()