import polars as pl
from time import perf_counter
# Load the dataset
start = perf_counter()
df = pl.read_ipc("ratings_ipc.ipc")

# Filter out ratings with status less than 1 or greater than 5

df = df.filter((df["status"] >= 1) & (df["status"] <= 5))

# Filter out items with less than 300 ratings

item_counts = df.groupby("anime_id").agg(pl.count("user").alias("item_rating_count"))

items_with_300_or_more_ratings = item_counts.filter(item_counts["item_rating_count"] >= 300)

df = df.join(items_with_300_or_more_ratings, left_on="anime_id", right_on="anime_id", how="inner")
# df = df.with_columns(
#     [
#         pl.col("user").count().over("anime_id").alias("item_rating_count")
#     ]
# ).filter(pl.col("item_rating_count") >= 300)


# Filter out users with less than 6 ratings

user_counts = df.groupby("user").agg(pl.count("anime_id").alias("rating_count"))

users_with_six_or_more_ratings = user_counts.filter(user_counts["rating_count"] >= 6)

df = df.join(users_with_six_or_more_ratings, left_on="user", right_on="user", how="inner")


# Filter out users whose .25 quantile rating is 0

user_quantiles = df.groupby("user").agg(pl.quantile("score", 0.25).alias("quantile_25"))

users_with_quantile_non_zero = user_quantiles.filter((user_quantiles["quantile_25"] != 0) & (user_quantiles["quantile_25"] != 10))

df = df.join(users_with_quantile_non_zero, left_on="user", right_on="user", how="inner")

median_scores = (

    df.groupby("user")

    .agg(pl.col("score").alias("user_median_score").median())

)

# Join the median scores with the original dataset

data_with_median = df.join(median_scores, on="user", how="left")

# Add the train_id column based on the conditions

df = (

    data_with_median

    .with_columns(

        pl.when(

            (pl.col("score") >= pl.col("user_median_score")) &

            (pl.col("status") >= 1) &

            (pl.col("status") <= 3)

        ).then(1).otherwise(0).alias("train_id")

    )

)
# Replace users with a unique number for each user

unique_usernames = df["user"].unique(maintain_order=True)

username_mapping = pl.DataFrame({

    "user": unique_usernames,

    "username": range(len(unique_usernames))

})

# Merge the dataset with the username mapping

df = df.join(username_mapping, on="user", how="left")

# Drop the "username" column

df = df.drop("user")

df = df.drop("user_median_score", "rating_count", "item_rating_count", "quantile_25")

print(perf_counter() - start)
print(df)
df.write_ipc("ratings_ipc_processed.ipc")