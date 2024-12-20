import pandas as pd

board_size = 4
n_turns_before_win = 2

n_games = 10000

path = f'hex_datasets/hex_{board_size}x{board_size}_{n_games}_games_{n_turns_before_win}_turns_before_win.csv'

df = pd.read_csv(path)

train_split = 0.8

train_end_index = int(len(df)*train_split)

print("Train end index:", train_end_index)

df_train = df[0:train_end_index]
df_test = df[train_end_index:]

set_train = set(map(tuple, df_train.to_numpy()))
set_test = set(map(tuple, df_test.to_numpy()))

# Intersect rows
intersection = set_train & set_test

# Union rows
union = set_train | set_test

# Difference rows
difference = set_train - set_test

# Pretty print the lengths
print({
    "Train set length": len(df_train),
    "Test set length": len(df_test),
    "Train set (unique rows)": len(set_train),
    "Test set (unique rows)": len(set_test),
    "Intersection": len(intersection),
    "Union": len(union),
    "Difference": len(difference)
})

columns_to_check = df.columns.difference(['winner'])
df_abs_sum = df[columns_to_check].abs().sum(axis=1)  # sum of absolute values per row
average_turns = df_abs_sum.mean()

print("Average number of turns across the games:", average_turns)

print(f"Average percent of game played: {average_turns/(average_turns + n_turns_before_win)*100:.1f}\%")