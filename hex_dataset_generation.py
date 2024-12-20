import numpy as np
import pandas as pd
import os
from numba import njit

@njit
def check_winner(board, player, board_dim, neighbors):
    visited = set()
    stack = []

    # Starting edges
    if player == 1:
        for j in range(board_dim):
            if board[0, j] == player:
                stack.append((0, j))
                visited.add((0, j))
    else:
        for i in range(board_dim):
            if board[i, 0] == player:
                stack.append((i, 0))
                visited.add((i, 0))

    while stack:
        i, j = stack.pop()
        if player == 1 and i == board_dim - 1:
            return True
        if player == -1 and j == board_dim - 1:
            return True

        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < board_dim and 0 <= nj < board_dim:
                if (ni, nj) not in visited and board[ni, nj] == player:
                    visited.add((ni, nj))
                    stack.append((ni, nj))

    return False

@njit
def simulate_game(board_dim, neighbors, n_turns):
    board = np.zeros((board_dim, board_dim), dtype=np.int8)
    open_positions = np.array([(i, j) for i in range(board_dim) for j in range(board_dim)], dtype=np.int32)
    np.random.shuffle(open_positions)  # Shuffle the open positions
    player = 1
    winner = 0

    max_moves = board_dim * board_dim
    # Initialize a circular buffer to store the last n_turns board states
    board_states = np.zeros((max(n_turns, 1), board_dim, board_dim), dtype=np.int8)
    state_index = 0  # Pointer to where to insert the next board state
    moves_made = 0  # Total moves made so far

    for pos in open_positions:
        i, j = pos
        board[i, j] = player

        if check_winner(board, player, board_dim, neighbors):
            winner = player
            break
        player *= -1  # Switch player

        # Store a copy of the current board state in the buffer
        board_states[state_index] = board.copy()
        state_index = (state_index + 1) % max(n_turns, 1)
        moves_made += 1

        
    if n_turns == 0:
        return board, winner
    else:
        if winner != 0:
            # Determine if we have enough history to return the board state n_turns before
            if moves_made >= n_turns:
                # Compute the index of the board state n_turns before the winner was found
                index = (state_index - n_turns) % n_turns
                board_n_turns_before = board_states[index]
            else:
                # Not enough moves made to have a board state n_turns before
                board_n_turns_before = board_states[0]  # Return the earliest board state
        else:
            # No winner found, return the last board state
            board_n_turns_before = board

        return board_n_turns_before, winner

def generate_hex_dataset(board_dim, n_games, n_turns_before_win, output_folder):
    neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    game_data_list = []
    for _ in range(4*n_games):
        board, winner = simulate_game(board_dim, np.array(neighbors, dtype=np.int32), n_turns_before_win)
        game_data = {}
        for i in range(board_dim):
            for j in range(board_dim):
                cell_label = f'cell{i}_{j}'
                game_data[cell_label] = board[i, j]
        game_data['winner'] = winner
        game_data_list.append(game_data)

    #Storage location
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'hex_{board_dim}x{board_dim}_{n_games}_games_{n_turns_before_win}_turns_before_win.csv')

    df = pd.DataFrame(game_data_list).drop_duplicates()


    #Sample each winner for equal class distribution
    df_p1_winner = df[df['winner'] == -1]
    df_p2_winner = df[df['winner'] == 1]

    df_p1_sample = df_p1_winner.sample(int(n_games/2), random_state=42)
    df_p2_sample = df_p2_winner.sample(int(n_games/2), random_state=42)

    df_balanced = pd.concat([df_p1_sample, df_p2_sample]).sample(frac=1).reset_index(drop=True)

    df_balanced.to_csv(output_file, index=False)
    print(f'Dataset saved to {output_file}')

# Example usage
for n in [4, 5, 6, 7]:
    generate_hex_dataset(
        board_dim=n,
        n_games=10000,
        n_turns_before_win=2,
        output_folder='hex_datasets'
    )