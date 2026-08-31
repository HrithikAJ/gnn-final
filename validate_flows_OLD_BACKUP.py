import pyarrow.parquet as pq
import pandas as pd

FILE_PATH = "data/intermediate/ugr16/cleaned_sorted.parquet"

parquet_file = pq.ParquetFile(FILE_PATH)
table = parquet_file.read()  # small enough now (15 columns) to read fully
df = table.to_pandas()

total = len(df)
print(f"Total rows loaded: {total:,}")
print("-" * 60)

# Build a boolean mask - starts as "all rows valid", then we knock out bad ones
valid_mask = pd.Series(True, index=df.index)

# Check 1: timestamp must not be null
check1 = df["timestamp"].notnull()
print(f"Check 1 - valid timestamp: {check1.sum():,} pass, {(~check1).sum():,} fail")
valid_mask &= check1

# Check 2: required identifying fields must not be null
check2 = df["source_id"].notnull() & df["destination_id"].notnull() & df["protocol"].notnull()
print(f"Check 2 - required fields present: {check2.sum():,} pass, {(~check2).sum():,} fail")
valid_mask &= check2

# Check 3: no negative packet/byte/duration values
check3 = (df["packet_count"] >= 0) & (df["byte_count"] >= 0) & (df["duration"] >= 0)
print(f"Check 3 - no negative values: {check3.sum():,} pass, {(~check3).sum():,} fail")
valid_mask &= check3

# Check 4: ports within valid range (0-65535), where present
check4 = df["source_port"].between(0, 65535) & df["destination_port"].between(0, 65535)
print(f"Check 4 - valid port range: {check4.sum():,} pass, {(~check4).sum():,} fail")
valid_mask &= check4

print("-" * 60)
valid_count = valid_mask.sum()
invalid_count = total - valid_count
print(f"TOTAL VALID FLOWS:   {valid_count:,} ({valid_count/total*100:.4f}%)")
print(f"TOTAL DROPPED FLOWS: {invalid_count:,} ({invalid_count/total*100:.4f}%)")

# Save only the valid rows as the next intermediate output
df_valid = df[valid_mask].reset_index(drop=True)
output_path = "data/intermediate/ugr16/validated_flows.parquet"
df_valid.to_parquet(output_path, index=False)
print(f"\nSaved validated flows to: {output_path}")