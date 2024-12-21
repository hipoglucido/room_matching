import random

import pandas as pd
from faker import Faker

fake = Faker()


def generate_advanced_room_name(base_name):
    prefixes = ["Deluxe", "Standard", "Luxury", "Executive", "Economy", "Premium"]
    suffixes = [
        "with ocean view", "near the pool", "with balcony", "pet-friendly",
        "for families", "with mountain view", "with garden access"
    ]
    modifiers = [
        "king bed", "queen bed", "twin beds", "spacious layout", "budget-friendly",
        "modern decor", "breakfast included", fake.city() + " style"
    ]

    # Randomly construct a complex name with combinations
    name_parts = [random.choice(prefixes), base_name]
    if random.random() > 0.5:
        name_parts.append(random.choice(suffixes))
    if random.random() > 0.5:
        name_parts.append(random.choice(modifiers))
    return " ".join(name_parts)


# Generate advanced variations for matching and non-matching logic
def generate_room_pair(base_name, is_match):
    room_a = generate_advanced_room_name(base_name)
    if is_match:
        # Modify room A slightly for room B (not a strict substring match)
        synonyms = {
            "king bed": "large bed", "queen bed": "double bed", "with balcony": "balcony access",
            "modern decor": "contemporary design", "spacious layout": "open space",
            "pet-friendly": "pets allowed", "budget-friendly": "affordable"
        }
        room_b = room_a
        for key, val in synonyms.items():
            if key in room_a:
                room_b = room_b.replace(key, val, 1)
                break
        # Randomize further order or add minor noise
        if random.random() > 0.5:
            room_b = room_b.replace(",", "").replace(" ", " ").strip()
    else:
        # Generate a completely different room name
        room_b = generate_advanced_room_name(base_name + " Alt")
    return room_a, room_b


def generate_synthetic_dataset(n_rows: int, match_ratio: float) -> pd.DataFrame:
    # Generate 200 rows of advanced synthetic data
    advanced_data = []
    for _ in range(n_rows):
        base_name = random.choice(["Room", "Suite", "Apartment", "Studio", "Villa"])
        is_match = random.random() < match_ratio
        room_a, room_b = generate_room_pair(base_name, is_match)
        advanced_data.append([room_a, room_b, is_match])

    # Create a DataFrame
    df = pd.DataFrame(advanced_data, columns=["A", "B", "match"]).drop_duplicates()
    return df
