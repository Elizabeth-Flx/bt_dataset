import pandas as pd
import numpy as np
import os

paths_ast = [
    "./astral/micro+nano_duplicates_ce_adjusted.parquet",
    "./astral/PXD046453_duplicates_ce_adjusted.parquet",
]

paths_tof = ["./timsTOF/tof_train.parquet"]

paths_pro = [
    "./proteome_tools/formatted/formatted_no_aug_test_0.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_0.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_1.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_2.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_3.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_4.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_5.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_6.parquet",
    "./proteome_tools/formatted/formatted_no_aug_train_7.parquet",
    "./proteome_tools/formatted/formatted_no_aug_val_0.parquet",
]

# read parquets and concatenate
df_ast = pd.concat([pd.read_parquet(path) for path in paths_ast])
print("Loaded Astral data")

df_tof = pd.concat([pd.read_parquet(path) for path in paths_tof])
print("Loaded timsTOF data")

df_pro = pd.concat([pd.read_parquet(path) for path in paths_pro])
print("Loaded ProteomeTools data")


dataset_cols = [
    'prosit_sequence',
    'charge',
    'collision_energy',
    'method_nr',
    'machine',
    'intensities_raw',
]

# set collision_energy_aligned_normed as collision_energy
df_tof['collision_energy'] = df_tof['collision_energy_aligned_normed']

# temporarily norm astral collision energy
df_ast['collision_energy'] = df_ast['collision_energy'] / 100

print("Collision energies set")

df_ast = df_ast[dataset_cols]
df_tof = df_tof[dataset_cols]
df_pro = df_pro[dataset_cols]
print("Columns selected")


df_ast = df_ast.sample(frac=1, random_state=42).reset_index(drop=True)
df_pro = df_pro.sample(frac=1, random_state=42).reset_index(drop=True)
df_tof = df_tof.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data shuffled")

df_combined = pd.concat([df_ast, df_tof, df_pro], axis=0, ignore_index=True)
df_combined
print("Data combined")


df_combined['intensities_raw'] = df_combined['intensities_raw'].apply(lambda x: x.astype(np.float64))
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data type (intensities) converted")

print(df_combined[df_combined['charge'] > 6])

# filter out charge > 6
df_combined = df_combined[df_combined['charge'] <= 6]
print("Charge filtered")

print(df_combined['charge'].value_counts())
print(df_combined['method_nr'].value_counts())
print(df_combined['machine'].value_counts())




method_map = {
    'CID': 0,
    'HCD': 1,
}

machine_map = {
    'Astral': 0,
    'TOF': 1,
    'Orbitrap_Fusion_Lumos': 2
}

df_combined['charge_oh']    = df_combined['charge']   .apply(lambda x: np.eye(6)[x-1]           .astype(int))
df_combined['method_nr_oh'] = df_combined['method_nr'].apply(lambda x: np.eye(2)[method_map[x]] .astype(int))
df_combined['machine_oh']   = df_combined['machine']  .apply(lambda x: np.eye(3)[machine_map[x]].astype(int))
print("One-hot encoding done")

df_combined['modified_sequence'] = df_combined['prosit_sequence']

df_combined.to_parquet("./full_dataset.parquet")
print("Full data saved")

df_combined.head(1_000_000).to_parquet("./reduced_dataset.parquet")
print("Reduced data saved")



# split 80/10/10

n_rows = df_combined.shape[0]

train   = df_combined.iloc[                 :int(n_rows * 0.8)]
val     = df_combined.iloc[int(n_rows * 0.8):int(n_rows * 0.9)]
test    = df_combined.iloc[int(n_rows * 0.9):]

train.to_parquet("./combined_dlomix_format_train.parquet")
print("Train data saved")
val  .to_parquet("./combined_dlomix_format_val.parquet")
print("Val data saved")
test .to_parquet("./combined_dlomix_format_test.parquet")
print("Test data saved")
