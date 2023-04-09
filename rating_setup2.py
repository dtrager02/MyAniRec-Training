import polars as pl
from time import perf_counter
# Load the dataset

start = perf_counter()
df = (pl.scan_ipc("ratings_ipc.ipc")
    .filter((pl.col("status") >= 1) & (pl.col("status") <= 5))
    .filter((pl.col("user").count().over("anime_id")) >= 300)
    .filter((pl.col("anime_id").count().over("user")) >= 6)
    .with_columns(
        pl.col("score").quantile(0.25).over("user").alias("user_quantile")
    )
    .filter((pl.col("user_quantile") > 0) & (pl.col("user_quantile") < 10))
    .drop("user_quantile")
    .with_columns(

        pl.when(

            (pl.col("score") >= pl.col("score").median().over("user")) &

            (pl.col("status") >= 1) &

            (pl.col("status") <= 3)

        ).then(1).otherwise(0).alias("train_id")

    ).collect()
)
unique_usernames = df["user"].unique(maintain_order=True)
username_mapping = pl.DataFrame({
    "user": unique_usernames,
    "username": range(len(unique_usernames))

})
# Merge the dataset with the username mapping
df = df.join(username_mapping, on="user", how="left")
# Drop the "username" column
df = df.drop("user")

print(perf_counter() - start)
print(df)