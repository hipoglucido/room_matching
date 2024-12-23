import random
from typing import Tuple

import pandas as pd
from faker import Faker
from loguru import logger
from random import choice, random, randrange


size_types = [
        ['small', 'petite', 'tiny', 'little', 'compact', 'miniature'],
        ['medium', 'normal', 'average', 'middling', 'moderate'],
        ['large', 'big', 'spacious', 'ample', 'vast', 'huge', 'enormous', 'gigantic', 'colossal', 'immense'],
        ['humungous', 'gargantuan', 'mammoth', 'tremendous']
    ]

properties = [
    ['balcony', 'terrace', 'patio', 'veranda'],
    ["well-equipped", "fully furnished", "all amenities", "modern conveniences"],
    ["nicely decorated", "beautifully designed", "stylishly decorated", "elegant", "chic", "modern", "classic",
     "rustic", "minimalist"],
    ['pet-friendly', 'pets allowed', 'pets welcome'],
    ["budget-friendly", "affordable", "cheap", "economical", "value"],
    ["luxury", "luxurious", "opulent", "grand", "premium"],
    ["cozy", "comfortable", "homey", "warm"],
    ["spacious", "roomy"],
    ["quiet", "peaceful", "tranquil", "serene"],
    ["convenient location", "central location", "close to attractions", "easy city access"],
    ["family-friendly", "kid-friendly", "suitable for children"],
    ["business-friendly", "executive", "work-ready", "with workspace"],
]
rooms = [
    ["studio", "open plan", "loft"],
    ["suite", "family suite",  "apartment"],
    ["executive suite", "suite"],
    ["room","double room"],
    ["room","queen room"],
    ["room","king room"],
    ["room","single room", "individual room"],
]
articles = ['a', 'the', *['' for _ in range(10)]]
modifiers = ['very', 'quite', 'super', *['' for _ in range(10)]]

def sample_and_append_element_to_matching_rooms(a, b, elements, could_be_empty:bool):
    elements_same_type = choice(elements)
    a.append(sample_element(elements_same_type, could_be_empty=could_be_empty))
    b.append(sample_element(elements_same_type, could_be_empty=could_be_empty))

def sample_element(elements_same_type, could_be_empty=bool):

    if could_be_empty:
        return choice(elements_same_type + ['' for _ in range(2*len(elements_same_type))])
    return choice(elements_same_type)


def generate_matching_rooms():
    a = [choice(articles),choice(modifiers)]
    b = [choice(articles),choice(modifiers)]
    sample_and_append_element_to_matching_rooms(a, b, size_types, could_be_empty=False)
    sample_and_append_element_to_matching_rooms(a, b, rooms, could_be_empty=False)
    sample_and_append_element_to_matching_rooms(a, b, properties, could_be_empty=True)
    sample_and_append_element_to_matching_rooms(a, b, properties, could_be_empty=True)
    a = add_perturbations(a)
    b = add_perturbations(b)
    return a, b

def add_perturbations(parts):


    new_parts = []
    for part in parts:
        new_part = part.strip()
        new_part = new_part.capitalize() if random() > 0.6 else part
        new_part = delete_random_character(new_part) if random() > 0.97 else new_part
        new_parts.append(new_part)
    result = " ".join(new_parts).strip()
    if random() > 0.9:
        result = result.upper()
    return result
def delete_random_character(text):
  """Deletes a random character from a string."""
  if text:
    position = randrange(len(text))
    return text[:position] + text[position+1:]
  return text
def generate_random_room():
    parts = [choice(articles),
             choice(modifiers),
             sample_element(choice(size_types), could_be_empty=False),
             sample_element(choice(rooms), could_be_empty=False),
             sample_element(choice(properties), could_be_empty=True),
             sample_element(choice(properties), could_be_empty=True),
             ]

    return add_perturbations(parts)

def generate_synthetic_dataset(n_rows, match_ratio):
    rows = []
    for _ in range(n_rows):
        match = random() < match_ratio
        if match:
            a, b = generate_matching_rooms()

        else:
            a = generate_random_room()
            b = generate_random_room()
        rows.append({"A": a, "B": b, "match": match})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for i in range(50):
        a,b = generate_matching_rooms()
        print(a,",",b)
    # for i in range(50):
    #     print(generate_random_room())
