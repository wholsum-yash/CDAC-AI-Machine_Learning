"""
Title: NBA Player Biographical & Birthday Paradox Analysis
Description: A modular Python exercise script for loading, sanitizing, and analyzing
             NBA player birth dates to explore shared birthdays and player lookups.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, datetime, and any required modules
import pandas as pd
from datetime import datetime


def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Data Discovery, Column Standardization & Sanitization (5 Marks)
    - Goal: Load the NBA player CSV file.
    - Standardize column names (remove whitespace, lowercase, snake_case).
    - Sanitize 'Birth Date' into standard datetime format.
    """
    print("\n[PHASE 1] Discovery & Data Sanitization (5 Marks)")
    print("-" * 65)
    # TODO: Load CSV file using pandas (handling potential quote formatting)
    # TODO: Standardize column headers (e.g., "Birth Date" -> "birth_date")
    # TODO: Parse 'birth_date' column into datetime objects using pd.to_datetime()
    # TODO: Inspect dataset shape and check for missing birth dates

    df = pd.read_csv("player_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    print("Shape \n", df.shape)
    print("Missing Bday Count \n", df["birth_date"].isna().sum())

    return df


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Distribution Analysis (10 Marks)
    - Goal: Analyze the distribution of birth dates across months and days.
    - Calculate and demonstrate the probability of shared birthdays (Birthday Paradox)
      within subsets of NBA players.
    """
    print("\n[PHASE 2] Birthday Paradox & Date Distribution (10 Marks)")
    print("-" * 65)
    # TODO: Extract month and day components from 'birth_date'
    # TODO: Count frequency of birthdays per calendar day
    # TODO: Identify dates shared by multiple NBA players in the dataset

    df["month"] = df["birth_date"].dt.month
    df["day"] = df["birth_date"].dt.day

    birthday_count = (
        df.groupby(["month", "day"]).size().reset_index(name="player_count")
    )

    birthday_count = birthday_count.sort_values(by="player_count", ascending=False)

    shared_birthdays = birthday_count[birthday_count["player_count"] > 1].sort_values(
        by="player_count", ascending=False
    )

    print("Birthday Freq: \n")
    print(birthday_count.sort_values(by="player_count", ascending=False))

    print("Birthdays shared by NBA players: \n", shared_birthdays)
    return df



    for i in df["player"]:
        if player_name in i:
            print(df[df["player"] == i][["pos", "ht", "wt", "colleges", "birth_date"]])
            return df[df["player"] == i]["birth_date"].iloc[0]


def phase_4_shared_birthday_cohort(df, player_name, target_dob):
    """
    [Phase 4] Shared Birthday Peer Finder (10 Marks)
    - Goal: Identify the target player's exact birth date.
    - List all other NBA players who share the exact same date of birth.
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort for '{player_name}' (10 Marks)")
    print("-" * 65)
    # TODO: Retrieve the birth date of the target player from Phase 3
    # TODO: Query the dataset for all other players born on that exact month and day (or year)
    # TODO: Output the list of peer players sharing the birthday

    target_month = target_dob.month
    target_day = target_dob.day
    target_year = target_dob.year

    cohort=df[((df['birth_date'].dt.month==target_month ) & (df['birth_date'].dt.day==target_day )) | (df['birth_date'].dt.year==target_year)]
    print(cohort[['player','colleges']])
    return cohort

def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Evaluation & Reporting
    - Goal: Summarize statistical findings on player longevity, birth months,
      and birthday coincidences.
    """
    print("\n[PHASE 5] Evaluation & Summary Reporting ")
    print("-" * 65)
    # TODO: Print final summary statistics regarding dataset coverage and duplicate birthdays
    
    print("Birth month freq: \n", df.groupby(['month']).size())
    print("-" * 65)
    print("Common Birth Month: \n", df['birth_date'].dt.month.mode())
    print("-" * 65)
    print("Common Birth Days: \n", df['birth_date'].mode())

# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================


def run_analytics_lifecycle():
    """
    Facade function coordinating the NBA analytics and birthday lookup workflow.
    """
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE ")
    print("=" * 65)

    # Target player to test lookup and shared birthdays
    target_player = "Kareem Abdul-Jabbar"

    # Executing phases chronologically
    df = phase_1_discovery_and_sanitization()
    df = phase_2_birthday_paradox_analysis(df)
    target_dob = phase_3_player_lookup_system(df, target_player)
    phase_4_shared_birthday_cohort(df, target_player, target_dob)
    phase_5_evaluation_and_summary(df)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()
