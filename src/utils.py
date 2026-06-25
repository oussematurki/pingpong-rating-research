import pandas as pd
import math


def create_player_base(df):
    players = pd.concat([df["Player1"], df["Player2"]]).unique()
    rows = [
        {
            "player": player,
            "rating": 1500.0,
            "matches_played": 0,
            "wins": 0,
            "losses": 0,
        }
        for player in players
    ]
    return pd.DataFrame(rows)


def update_rating(r_old_A, r_old_B, actual_winner, metric_type, match_data, K):
    # Step 1: Expected Probability
    E_A = 1 / (1 + 10 ** ((r_old_B - r_old_A) / 400))
    E_B = 1 - E_A

    S_A = 1 if actual_winner == "A" else 0
    S_B = 1 - S_A

    # Identify pre-match rating gap from perspective of the winner
    r_win = r_old_A if actual_winner == "A" else r_old_B
    r_lose = r_old_B if actual_winner == "A" else r_old_A
    rating_diff = r_win - r_lose

    # Step 2: Calculate Multiplier based on your test branch
    if metric_type == "set_based":
        # match_data = {"sets_won": X, "sets_lost": Y}
        set_diff = match_data["sets_won"] - match_data["sets_lost"]
        multiplier = math.log(set_diff + 1) * (2.2 / (rating_diff * 0.001 + 2.2))

    elif metric_type == "points_based":
        # match_data = {"point_diff": Z}
        pt_diff = match_data["point_diff"]
        multiplier = ((pt_diff + 3) ** 0.8) / (7.5 + 0.006 * rating_diff)

    else:
        multiplier = 1.0  # Standard Elo baseline

    # Step 3: Apply changes
    r_new_A = r_old_A + K * multiplier * (S_A - E_A)
    r_new_B = r_old_B + K * multiplier * (S_B - E_B)

    return round(r_new_A, 2), round(r_new_B, 2), round(E_A, 4), round(E_B, 4)


def safe_sum(*args):
    return sum(x for x in args if pd.notna(x))
