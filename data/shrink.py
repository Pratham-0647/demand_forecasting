import pandas as pd
for f in ["sales_train_evaluation", "calendar", "sell_prices"]:
    df = pd.read_csv(f + ".csv")
    for c in df.select_dtypes("int64"):
        df[c] = pd.to_numeric(df[c], downcast="integer")
    df.to_parquet(f + ".parquet", compression="zstd")
    print(f, df.shape)
