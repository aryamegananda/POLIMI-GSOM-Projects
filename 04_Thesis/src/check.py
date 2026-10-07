import filecmp
from prepare_data import DATASETS, CONDITIONS

for ds_name in DATASETS:
    for cond_name in CONDITIONS:
        a = f"data/messy_run1/{ds_name}/{cond_name}/train.csv"
        b = f"data/messy/{ds_name}/{cond_name}/train.csv"
        same = filecmp.cmp(a, b, shallow=False)
        print(ds_name, cond_name, "identical" if same else "DIFFERENT")