import numpy as np

# Poisson function for calculating probabilities
def poisson_prob(lmbda, k):
    return (lmbda**k * np.exp(-lmbda)) / np.math.factorial(k)

# New function to calculate adjusted goals
def calculate_adjusted_goals(xG, G, over_under_perf_ratio, w1=0.7, w2=0.3, adj_factor=0.2):
    adjusted_goals = ((w1 * xG) + (w2 * G) + ((G / xG - 1) * adj_factor * xG))
    return max(0, adjusted_goals)

# Predict goals based on xG and actual G
def predict_goals(home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games):
    home_attack_ratio = home_G / home_xG if home_xG > 0 else 1
    away_attack_ratio = away_G / away_xG if away_xG > 0 else 1
    home_defense_ratio = home_GA / home_xGA if home_xGA > 0 else 1
    away_defense_ratio = away_G / away_xGA if away_xGA > 0 else 1
    
    expected_home_goals = calculate_adjusted_goals((home_xG / home_games), (home_G / home_games), home_attack_ratio)
    expected_away_goals = calculate_adjusted_goals((away_xG / away_games), (away_G / away_games), away_attack_ratio)
    
    # Predicted Home & Away goals using xG
    predicted_home_goals_xG = expected_home_goals
    predicted_away_goals_xG = expected_away_goals
    predicted_total_goals_xG = ((predicted_home_goals_xG + predicted_away_goals_xG))
    
    # Predicted Home & Away goals using G
    predicted_home_goals_G = (home_G / home_games) * (away_GA / away_games)
    predicted_away_goals_G = (away_G / away_games) * (home_GA / home_games)
    predicted_total_goals_G = predicted_home_goals_G + predicted_away_goals_G
    predicted_total_goals_G_xG = predicted_total_goals_xG -  predicted_total_goals_G 
    if predicted_total_goals_G > predicted_total_goals_xG:
        predicted_total_goals_G_xG = predicted_total_goals_G - predicted_total_goals_xG
    
    return predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G, predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG

# Simulate match outcomes
def simulate_match(home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games, n_simulations=1000000):
    predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G, predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG  = predict_goals(
        home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games
    )
    
    home_goals = np.random.poisson(predicted_home_goals_xG, n_simulations)
    away_goals = np.random.poisson(predicted_away_goals_xG, n_simulations)
    total_goals = home_goals + away_goals
    
    return predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G, home_goals, away_goals, total_goals, predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG

# Calculate probabilities based on simulated goals
def calculate_probabilities(team_goals, min_prob=0.10, max_prob=0.98):
    over_0_5 = np.clip(np.mean(team_goals > 0.5), min_prob, max_prob)
    under_0_5 = np.clip(np.mean(team_goals <= 0.5), min_prob, max_prob)
    over_1_5 = np.clip(np.mean(team_goals > 1.5), min_prob, max_prob)
    under_1_5 = np.clip(np.mean(team_goals <= 1.5), min_prob, max_prob)
    over_2_5 = np.clip(np.mean(team_goals > 2.5), min_prob, max_prob)
    under_2_5 = np.clip(np.mean(team_goals <= 2.5), min_prob, max_prob)
    over_3_5 = np.clip(np.mean(team_goals > 3.5), min_prob, max_prob)
    under_3_5 = np.clip(np.mean(team_goals <= 3.5), min_prob, max_prob)

    return {
        "Over 0.5": over_0_5, "Under 0.5": under_0_5,
        "Over 1.5": over_1_5, "Under 1.5": under_1_5,
        "Over 2.5": over_2_5, "Under 2.5": under_2_5,
        "Over 3.5": over_3_5, "Under 3.5": under_3_5
    }

# Calculate odds from probabilities
def calculate_odds(probabilities):
    odds = {outcome: 1 / prob if prob > 0 else float('inf') for outcome, prob in probabilities.items()}
    return odds

# Final match prediction function
def predict_match(home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games, n_simulations=1000000):
    predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G, home_goals, away_goals, total_goals, predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG = simulate_match(
        home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games,  n_simulations
    )

    # Calculate probabilities and odds for total goals
    total_probabilities = calculate_probabilities(total_goals)
    total_odds = calculate_odds(total_probabilities)

    # Calculate probabilities and odds for home goals
    home_probabilities = calculate_probabilities(home_goals)
    home_odds = calculate_odds(home_probabilities)

    # Calculate probabilities and odds for away goals
    away_probabilities = calculate_probabilities(away_goals)
    away_odds = calculate_odds(away_probabilities)

    return (total_probabilities, total_odds, home_probabilities, home_odds, away_probabilities, away_odds, 
            predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G, 
            predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG)

# Example input data for testing
home_games = 1
home_G = 3     # Actual goals scored by home team in previous match
home_GA = 1    # Actual goals conceded by home team
home_xG = 2.4  # Home team expected goals
home_xGA = 0.9 # Home team expected goals against (xGA)

away_games = 1
away_G = 2     # Actual goals scored by away team in previous match
away_GA = 3    # Actual goals conceded by away team
away_xG = 0.1  # Away team expected goals
away_xGA = 2.7 # Away team expected goals against (xGA)
total_avg_goals_per_match = ((home_G + away_G) / (home_games + away_games))

# Predict match outcomes
(total_probabilities, total_odds, home_probabilities, home_odds, away_probabilities, away_odds,
 predicted_home_goals_xG, predicted_away_goals_xG, predicted_home_goals_G, predicted_away_goals_G,
 predicted_total_goals_G, predicted_total_goals_xG, predicted_total_goals_G_xG) = predict_match(
    home_xG, away_xG, home_G, away_G, home_xGA, away_xGA, home_GA, away_GA, home_games, away_games
)

# Print probabilities and odds
print("Match Prediction Probabilities and Odds:\n")
# Print all Over probabilities
print("Over Probabilities:")
for outcome in ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5"]:
    print(f"{outcome}: Probability: {probabilities[outcome]:.2%}, Odds: {odds[outcome]:.2f}")
print()  # Blank line
# Print all Under probabilities
print("Under Probabilities:")
for outcome in ["Under 0.5", "Under 1.5", "Under 2.5", "Under 3.5"]:
    print(f"{outcome}: Probability: {probabilities[outcome]:.2%}, Odds: {odds[outcome]:.2f}")
print()

# Print all Over probabilities for home goals
print("\nHome Goals Over Probabilities:")
for outcome in ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5"]:
   print(f"{outcome}: Probability = {home_probabilities[outcome]:.2%}, Odds = {home_odds[outcome]:.2f}")
print()  # Blank line
# Print all Under probabilities for home goals
print("\nHome Goals Under Probabilities:")
for outcome in ["Under 0.5", "Under 1.5", "Under 2.5", "Under 3.5"]:
   print(f"{outcome}: Probability = {home_probabilities[outcome]:.2%}, Odds = {home_odds[outcome]:.2f}")

print()  # Blank line

# Print all Over probabilities for away goals
print("\nAway Goals Over Probabilities:")
for outcome in ["Over 0.5", "Over 1.5", "Over 2.5", "Over 3.5"]:
   print(f"{outcome}: Probability = {away_probabilities[outcome]:.2%}, Odds = {away_odds[outcome]:.2f}")
print()  # Blank line
# Print all Under probabilities for away goals
print("\nAway Goals Under Probabilities:")
for outcome in ["Under 0.5", "Under 1.5", "Under 2.5", "Under 3.5"]:
   print(f"{outcome}: Probability = {away_probabilities[outcome]:.2%}, Odds = {away_odds[outcome]:.2f}")

print()  # Blank line
    
print("Expected / Predicted Goals based on xG:")
print(f"Predicted Home Goals (xG): {predicted_home_goals_xG:.2f}")
print(f"Predicted Away Goals (xG): {predicted_away_goals_xG:.2f}")
print(f"Predicted Total Goals (xG): {predicted_total_goals_xG:.2f}")
print()

print("Expected / Predicted Goals Based on G:")
print(f"Predicted Home Goals (G): {predicted_home_goals_G:.2f}")
print(f"Predicted Away Goals (G): {predicted_away_goals_G:.2f}")
print(f"Predicted Total Goals (G): {predicted_total_goals_G:.2f}")
print()


print("Expected Total Goals / Predicted Total Goals Based on G & xG difference:")
print(f"Predicted Goals (G): {predicted_total_goals_G_xG:.2f}")