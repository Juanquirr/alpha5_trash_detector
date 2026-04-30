"""
Per-class prompt definitions for SAM3 trash auto-labeling.

Priority: lower number wins cross-class NMS.
  1 = most specific (can)
  5 = catch-all (trash)
"""

CLASS_DEFS: dict[int, dict] = {
    0: {
        "name": "plastic_bottle",
        "priority": 2,
        "prompts": [
            "plastic bottle with cap",
            "plastic water bottle",
            "plastic container with lid",
            "clear plastic bottle",
            "plastic soda bottle",
            "plastic detergent bottle",
            "plastic juice bottle",
            "plastic shampoo bottle",
            "plastic jug with lid",
            "PET plastic bottle",
        ],
    },
    1: {
        "name": "glass",
        "priority": 2,
        "prompts": [
            "glass bottle",
            "beer bottle",
            "wine bottle",
            "brown glass bottle",
            "green glass bottle",
            "empty glass bottle",
            "glass beer bottle",
            "glass wine bottle",
            "glass liquor bottle",
        ],
    },
    2: {
        "name": "can",
        "priority": 1,
        "prompts": [
            "aluminum can",
            "soda can",
            "beer can",
            "metal beverage can",
            "crushed aluminum can",
            "tin can",
            "energy drink can",
            "empty metal can",
        ],
    },
    3: {
        "name": "plastic_bag",
        "priority": 3,
        "prompts": [
            "plastic bag",
            "grocery plastic bag",
            "trash bag",
            "garbage bag",
            "black plastic bag",
            "white plastic bag",
            "large crumpled plastic bag",
            "shopping plastic bag",
            "clear plastic bag",
        ],
    },
    4: {
        "name": "metal_scrap",
        "priority": 2,
        "prompts": [
            "tuna can",
            "opened tin can",
            "aerosol spray can",
            "deodorant spray can",
            "spray paint can",
            "aluminum foil",
            "crumpled aluminum foil",
            "metal food can",
            "metal lid",
            "tin lid",
        ],
    },
    5: {
        "name": "plastic_wrapper",
        "priority": 3,
        "prompts": [
            "chip bag",
            "crisp packet",
            "candy wrapper",
            "chocolate wrapper",
            "snack wrapper",
            "food plastic wrapper",
            "crinkled plastic wrapper",
            "plastic snack packaging",
            "plastic food packaging",
        ],
    },
    6: {
        "name": "trash_pile",
        "priority": 4,
        "prompts": [
            "pile of trash",
            "garbage pile",
            "heap of litter",
            "trash heap",
            "accumulated garbage",
            "mound of rubbish",
            "mixed waste pile",
            "scattered waste",
        ],
    },
    7: {
        "name": "trash",
        "priority": 5,
        "prompts": [
            "trash",
            "litter",
            "garbage",
            "debris",
            "rubbish",
            "discarded waste",
            "waste",
            "abandoned waste",
        ],
    },
}
