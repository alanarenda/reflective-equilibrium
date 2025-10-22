import pandas as pd
from datasets import load_dataset


def prepare_govreport_data():
    # Load both datasets
    print("Loading datasets...")
    ds_qs = load_dataset("launch/gov_report_qs", "document")
    ds_gr = load_dataset("launch/gov_report", trust_remote_code=True)

    # Convert gov_report_qs splits to pandas
    train_qs = ds_qs["train"].to_pandas()
    val_qs = ds_qs["validation"].to_pandas()
    test_qs = ds_qs["test"].to_pandas()

    # Convert gov_report splits to pandas and combine ALL splits
    train_gr = ds_gr["train"].to_pandas()
    val_gr = ds_gr["validation"].to_pandas()
    test_gr = ds_gr["test"].to_pandas()
    all_gr = pd.concat([train_gr, val_gr, test_gr], ignore_index=True)

    # Join each split of gov_report_qs against the full gov_report dataset
    train_combined = train_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')
    val_combined = val_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')
    test_combined = test_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')

    with open('train_combined.csv', 'w') as f:
        train_combined.to_csv(f, index=False)

    with open('val_combined.csv', 'w') as f:
        val_combined.to_csv(f, index=False) 

    with open('test_combined.csv', 'w') as f:
        test_combined.to_csv(f, index=False)    
    

if __name__ == "__main__":
    prepare_govreport_data()