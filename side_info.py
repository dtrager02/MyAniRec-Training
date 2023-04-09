import polars as pl

# Read the dataset (assuming it's a CSV file)

df = pl.read_ipc("ratings_ipc_processed.ipc")

# Create a separate dataframe with 1 row per user

user_stats = (

    df.groupby("username")

    .agg(

        [

            pl.col("created_at").mean().alias("mean_created_at"),

            pl.col("created_at").std().alias("std_created_at"),

            (pl.col("created_at").max() - pl.col("created_at").min()).alias("range_created_at"),

            pl.col("created_at").sort().diff().mean().alias("avg_diff_sorted_created_at"),

            pl.col("username").count().alias("total_ratings"),

            pl.col("score").mean().alias("average_score"),

            pl.col("score").std().alias("std_score"),

            pl.col("score").min().alias("min_score"),
        ]

    )

)

print(user_stats)
user_stats.write_ipc("user_stats.ipc")