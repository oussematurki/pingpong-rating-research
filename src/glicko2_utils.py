"""
glicko2_utils.py

Glicko-2 implementation for pingpong-rating-research.
Mirrors the conventions in src/utils.py:
  - create_glicko_player_base() parallels create_player_base()
  - ratings stored as float, matches_played tracked per player
  - predict-before-update pattern preserved (no leakage)

Reference: Mark Glickman, "Example of the Glicko-2 system"
http://www.glicko.net/glicko/glicko2.pdf
"""

import math
import pandas as pd

# ---- Glicko-2 constants -----------------------------------------------
GLICKO_SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5          # system constant, constrains volatility change (0.3-1.2 typical)
EPSILON = 0.000001  # convergence tolerance for volatility iteration


# ---- Player base --------------------------------------------------------

def create_glicko_player_base(df):
    """
    Same shape/convention as create_player_base() in utils.py -- a DataFrame
    with one row per player -- just with two extra Glicko-2 fields (rd, vol).

        player | rating | rd  | vol  | matches_played

    Use player_base_to_dict() / dict_to_player_base() below to move in and out
    of a dict for the O(1)-lookup update loop, then hand back a DataFrame in
    the same shape create_player_base() would give you.
    """
    players = pd.concat([df["Player1"], df["Player2"]]).unique()
    rows = [
        {
            "player": player,
            "rating": DEFAULT_RATING,
            "rd": DEFAULT_RD,
            "vol": DEFAULT_VOL,
            "matches_played": 0,
        }
        for player in players
    ]
    return pd.DataFrame(rows)


def player_base_to_dict(player_base_df):
    """DataFrame (as above) -> dict keyed by player name, for fast in-loop lookups."""
    return {
        row["player"]: {
            "rating": row["rating"],
            "rd": row["rd"],
            "vol": row["vol"],
            "matches_played": row["matches_played"],
        }
        for _, row in player_base_df.iterrows()
    }


def dict_to_player_base(player_dict):
    """Inverse of player_base_to_dict() -- back to the DataFrame shape."""
    rows = [
        {"player": player, **state}
        for player, state in player_dict.items()
    ]
    return pd.DataFrame(rows)



# ---- Scale conversion ----------------------------------------------------

def to_glicko2_scale(rating, rd):
    mu = (rating - DEFAULT_RATING) / GLICKO_SCALE
    phi = rd / GLICKO_SCALE
    return mu, phi


def from_glicko2_scale(mu, phi):
    rating = mu * GLICKO_SCALE + DEFAULT_RATING
    rd = phi * GLICKO_SCALE
    return rating, rd


# ---- Core Glicko-2 math --------------------------------------------------

def g(phi):
    return 1 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)


def E(mu, mu_j, phi_j):
    """Expected score of a player (mu) against opponent (mu_j, phi_j)."""
    return 1 / (1 + math.exp(-g(phi_j) * (mu - mu_j)))


def predict_probability(rating_a, rd_a, rating_b, rd_b):
    """
    Win probability for player A vs player B, on the original (1500-based) scale.
    Use this BEFORE applying any update, exactly like win_probability_p1 in the
    Elo notebooks — predict first, then update ratings.
    """
    mu_a, phi_a = to_glicko2_scale(rating_a, rd_a)
    mu_b, phi_b = to_glicko2_scale(rating_b, rd_b)
    return E(mu_a, mu_b, phi_b)


def _new_volatility(phi, delta, v, sigma):
    """Illinois algorithm (regula falsi variant) to solve for sigma'."""
    a = math.log(sigma ** 2)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2 * (phi ** 2 + v + ex) ** 2
        return (num / den) - ((x - a) / TAU ** 2)

    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    fA, fB = f(A), f(B)
    while abs(B - A) > EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA = fA / 2
        B, fB = C, fC

    return math.exp(A / 2)


def update_player_one_period(rating, rd, vol, opponents):
    """
    Update ONE player's (rating, rd, vol) after a rating period, given a list of
    results against opponents faced in that period.

    opponents: list of (opp_rating, opp_rd, score) with score in {0, 0.5, 1}.
    If opponents is empty (player was inactive that period), only RD inflates
    (step 6 of the Glicko-2 paper), rating and volatility stay put.

    Returns (new_rating, new_rd, new_vol).
    """
    mu, phi = to_glicko2_scale(rating, rd)

    if not opponents:
        phi_star = math.sqrt(phi ** 2 + vol ** 2)
        new_rating, new_rd = from_glicko2_scale(mu, phi_star)
        return new_rating, new_rd, vol

    g_list, E_list, s_list = [], [], []
    for opp_rating, opp_rd, score in opponents:
        mu_j, phi_j = to_glicko2_scale(opp_rating, opp_rd)
        g_j = g(phi_j)
        E_j = E(mu, mu_j, phi_j)
        g_list.append(g_j)
        E_list.append(E_j)
        s_list.append(score)

    v_inv = sum(g_j ** 2 * E_j * (1 - E_j) for g_j, E_j in zip(g_list, E_list))
    v = 1 / v_inv

    delta = v * sum(g_j * (s_j - E_j) for g_j, s_j, E_j in zip(g_list, s_list, E_list))

    new_vol = _new_volatility(phi, delta, v, vol)

    phi_star = math.sqrt(phi ** 2 + new_vol ** 2)
    new_phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
    new_mu = mu + new_phi ** 2 * sum(g_j * (s_j - E_j) for g_j, s_j, E_j in zip(g_list, s_list, E_list))

    new_rating, new_rd = from_glicko2_scale(new_mu, new_phi)
    return new_rating, new_rd, new_vol