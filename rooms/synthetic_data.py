import random
from typing import List, Tuple
import pandas as pd

from random import choice, random, randrange, seed
import numpy as np

from rooms.constants_config import SEED

# List of size types for rooms
size_types: List[List[str]] = [
    ["small", "petite", "tiny", "little", "compact", "miniature"],
    ["medium", "normal", "average", "middling", "moderate"],
    [
        "large",
        "big",
        "spacious",
        "ample",
        "vast",
        "huge",
        "enormous",
        "gigantic",
        "colossal",
        "immense",
    ],
    ["humungous", "gargantuan", "mammoth", "tremendous"],
]

# List of properties for rooms
properties: List[List[str]] = [
    ["balcony", "terrace", "patio", "veranda"],
    ["well-equipped", "fully furnished", "all amenities", "modern conveniences"],
    [
        "nicely decorated",
        "beautifully designed",
        "stylishly decorated",
        "elegant",
        "chic",
        "modern",
        "classic",
    ],
    ["pet-friendly", "pets allowed", "pets welcome"],
    ["budget-friendly", "affordable", "cheap", "economical", "value"],
    ["luxury", "luxurious", "opulent", "grand", "premium"],
    ["cozy", "comfortable", "homey", "warm"],
    ["spacious", "roomy"],
    ["quiet", "peaceful", "tranquil", "serene"],
    [
        "convenient location",
        "central location",
        "close to attractions",
        "easy city access",
    ],
    ["family-friendly", "kid-friendly", "suitable for children"],
    ["business-friendly", "executive", "work-ready", "with workspace"],
    ["rustic", "minimalist"],
]

# List of room types
rooms: List[List[str]] = [
    ["studio", "open plan", "loft"],
    ["suite", "family suite", "apartment"],
    ["executive suite", "suite"],
    ["room", "double room"],
    ["room", "queen room"],
    ["room", "king room"],
    ["room", "single room", "individual room"],
]

# List of articles and modifiers
articles: List[str] = ["a", "the", *["" for _ in range(10)]]
modifiers: List[str] = ["very", "quite", "super", *["" for _ in range(10)]]


def sample_and_append_element_to_matching_rooms(
    a: List[str], b: List[str], elements: List[List[str]], could_be_empty: bool
) -> None:
    """
    Samples an element from the given list and appends it to both lists a and b.

    Args:
        a (list): First list to append the sampled element.
        b (list): Second list to append the sampled element.
        elements (list): List of elements to sample from.
        could_be_empty (bool): Flag indicating if the sampled element can be empty.
    """
    elements_same_type = choice(elements)
    a.append(sample_element(elements_same_type, could_be_empty=could_be_empty))
    b.append(sample_element(elements_same_type, could_be_empty=could_be_empty))


def sample_element(elements_same_type: List[str], could_be_empty: bool = False) -> str:
    """
    Samples an element from the given list, with an option to include empty elements.

    Args:
        elements_same_type (list): List of elements to sample from.
        could_be_empty (bool): Flag indicating if the sampled element can be empty.

    Returns:
        str: Sampled element.
    """
    if could_be_empty:
        return choice(
            elements_same_type + ["" for _ in range(2 * len(elements_same_type))]
        )
    return choice(elements_same_type)


def generate_matching_rooms() -> Tuple[str, str]:
    """
    Generates two matching room descriptions with slight variations.

    Returns:
        Tuple[str, str]: Two matching room descriptions.
    """
    a: List[str] = [choice(articles), choice(modifiers)]
    b: List[str] = [choice(articles), choice(modifiers)]
    sample_and_append_element_to_matching_rooms(a, b, size_types, could_be_empty=False)
    sample_and_append_element_to_matching_rooms(a, b, rooms, could_be_empty=False)
    sample_and_append_element_to_matching_rooms(a, b, properties, could_be_empty=True)
    sample_and_append_element_to_matching_rooms(a, b, properties, could_be_empty=True)
    a = add_perturbations(a)
    b = add_perturbations(b)
    return a, b


def add_perturbations(parts: List[str]) -> str:
    """
    Adds random perturbations to the given list of parts.

    Args:
        parts (list): List of parts to perturb.

    Returns:
        str: Perturbed string.
    """
    new_parts: List[str] = []
    for part in parts:
        new_part = part.strip()
        new_part = new_part.capitalize() if random() > 0.6 else part
        new_part = delete_random_character(new_part) if random() > 0.97 else new_part
        new_parts.append(new_part)
    result = " ".join(new_parts).strip()
    if random() > 0.9:
        result = result.upper()
    return result


def delete_random_character(text: str) -> str:
    """
    Deletes a random character from the given string.

    Args:
        text (str): String to delete a character from.

    Returns:
        str: String with a random character deleted.
    """
    if text:
        position = randrange(len(text))
        return text[:position] + text[position + 1 :]
    return text


def generate_random_room() -> str:
    """
    Generates a random room description.

    Returns:
        str: Random room description.
    """
    parts: List[str] = [
        choice(articles),
        choice(modifiers),
        sample_element(choice(size_types), could_be_empty=False),
        sample_element(choice(rooms), could_be_empty=False),
        sample_element(choice(properties), could_be_empty=True),
        sample_element(choice(properties), could_be_empty=True),
    ]
    return add_perturbations(parts)


def set_seed():
    # Set a seed for random
    seed(SEED)

    # Set a seed for NumPy
    np.random.seed(SEED)


def generate_synthetic_dataset(n_rows: int, match_ratio: float) -> pd.DataFrame:
    """
    Generates a synthetic dataset of room descriptions.

    Args:
        n_rows (int): Number of rows in the dataset.
        match_ratio (float): Ratio of matching room descriptions.

    Returns:
        pd.DataFrame: Synthetic dataset of room descriptions.
    """
    set_seed()
    rows: List[dict] = []
    for _ in range(n_rows):
        match = random() < match_ratio
        if match:
            a, b = generate_matching_rooms()
        else:
            a = generate_random_room()
            b = generate_random_room()
        rows.append({"A": a, "B": b, "match": match})
    return pd.DataFrame(rows)
