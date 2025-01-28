import json
import random
from data.database import get_all_areas, get_all_ingredients, get_all_categories
from pipeline import get_args, process_nlu, update_nlu_slots
from recipe_state_tracker import RecipeStateTracker
from utils import load_model
from collections import Counter
from typing import List, Dict

TEMPLATES = [
    "What can I cook with {ingredient}?",
    "Can you suggest a recipe with {ingredient1} and {ingredient2}?",
    "Show me a {nationality1} dish I can prepare.",
    "What are some recipes that include {ingredient}?",
    "Can I make something with {ingredient1} and {ingredient2}?",
    "Suggest a {category} recipe with {ingredient} as ingredient.",
    "What's a good {category} recipe for {nationality1} cuisine?",
    "What's a classic {nationality1} recipe?",
    "What {category} recipes do you have?",
    "Do you have a traditional {nationality1} recipe with {ingredient}?",
    "What can I cook for breakfast using {ingredient}?",
    "What's a popular {nationality1} dish I can cook?",
    "Can you suggest a recipe comes from {nationality1} or {nationality2} cuisines?",
]


def calculate_nlu_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate precision, recall, and F1 score for intents and slots
    
    Args:
        predictions (List[Dict]): List of dictionaries containing intent and slot predictions
        
    Returns:
        Dict: Dictionary containing metrics for both intents and slots
    """
    def calculate_metrics(tp: int, fp: int, fn: int) -> Dict:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }

    # Initialize counters
    intent_counts = Counter()
    slot_counts = Counter()

    for pred in predictions:
        # Intent evaluation
        true_intents = set(pred["intent"])
        if "detected_intent" not in pred:
            continue
        pred_intents = set(pred["detected_intent"])
        
        intent_counts["tp"] += len(true_intents & pred_intents)  # intersection
        intent_counts["fp"] += len(pred_intents - true_intents)  # false predictions
        intent_counts["fn"] += len(true_intents - pred_intents)  # missed predictions

        # Slot evaluation
        true_slots = {
            slot: {str(value).lower().replace(" ", "") for value in values} 
            if isinstance(values, list) else 
            {str(values).lower().replace(" ", "")} if values is not None else set()
            for slot, values in pred["slots"].items()
        }
        
        pred_slots = {
            slot: {str(value).lower().replace(" ", "") for value in values}
            if isinstance(values, list) else
            {str(values).lower().replace(" ", "")} if values is not None else set()
            for slot, values in pred["detected_slots"].items()
        }

        # Count slot matches
        for slot in set(true_slots.keys()) | set(pred_slots.keys()):
            true_values = true_slots.get(slot, set())
            pred_values = pred_slots.get(slot, set())

            if len(pred_values - true_values) != 0:
                print(f"False positive: {pred_values - true_values}")
                print(f"Predicted: {pred}")

            if len(true_values - pred_values) != 0:
                print(f"False negative: {true_values - pred_values}")
            
            slot_counts["tp"] += len(true_values & pred_values)  # intersection
            slot_counts["fp"] += len(pred_values - true_values)  # false predictions
            slot_counts["fn"] += len(true_values - pred_values)  # missed predictions

    # Calculate metrics
    intent_metrics = calculate_metrics(
        intent_counts["tp"],
        intent_counts["fp"],
        intent_counts["fn"]
    )
    
    slot_metrics = calculate_metrics(
        slot_counts["tp"],
        slot_counts["fp"],
        slot_counts["fn"]
    )

    return {
        "intent_metrics": {
            **intent_metrics,
            "counts": dict(intent_counts)
        },
        "slot_metrics": {
            **slot_metrics,
            "counts": dict(slot_counts)
        }
    }

# Function to fill a template with provided slot values
def fill_template(template, slots):
    try:
        return template.format(**slots)
    except KeyError as e:
        return f"Missing slot value for {e}"

# Function to generate filled questions with slots and produce answers
def generate_filled_questions(templates, all_ingredients, all_nationalities, all_categories, num_questions=10):
    all_questions = []

    for _ in range(num_questions):
        # Randomly select values for the slots
        slots = {
            "ingredient": random.choice(all_ingredients),
            "ingredient1": random.choice(all_ingredients),
            "ingredient2": random.choice(all_ingredients),
            "nationality1": random.choice(all_nationalities),
            "nationality2": random.choice(all_nationalities),
            "category": random.choice(all_categories)
        }

        # Generate questions and populate the answer format
        for template in templates:
            question = fill_template(template, slots)
            answer = {
                "intent": ['recipe_recommendation'],
                "slots": {
                    "ingredients": [],
                    "nationality": [],
                    "category": []
                },
                "question": question
            }
            if "{ingredient}" in template:
                answer["slots"]["ingredients"] = [slots["ingredient"]]

            if "{ingredient1}" in template and "{ingredient2}" in template:
                answer["slots"]["ingredients"] = [slots["ingredient1"], slots["ingredient2"]]
            
                
            if "{nationality1}" in template:
                answer["slots"]["nationality"] = [slots["nationality1"]]
            if "{nationality2}" in template:
                answer["slots"]["nationality"] = [slots["nationality1"], slots["nationality2"]]
            if "{category}" in template:
                answer["slots"]["category"] = [slots["category"]]
            
            # Remove None values in the ingredients list
            all_questions.append(answer)

    return all_questions

# Example usage
if __name__ == "__main__":
    all_ingredients = get_all_ingredients()  # Example: ["chicken", "tomatoes", "basil"]
    all_nationalities = get_all_areas()  # Example: ["Italian", "Indian", "Mexican"]
    all_categories = get_all_categories()  # Example: ["main course", "dessert", "soup"]

    test_data_recipe_reccomendation = generate_filled_questions(TEMPLATES, all_ingredients, all_nationalities, all_categories, num_questions=10)

    args = get_args()
    model, tokenizer = load_model(args)
    state_tracker = RecipeStateTracker()
    
    compute_metrics = False
    if compute_metrics:
    # Print all generated answers
        for item in test_data_recipe_reccomendation:
            user_input = item["question"]
            intents = process_nlu(user_input, state_tracker, [], model, tokenizer, args)
            item["detected_intent"] = intents
            nlu = {"intent": "recipe_recommendation", "slots": {}}
            update_nlu_slots(nlu, user_input, state_tracker, model, tokenizer, args)  
            if "nationality" not in nlu["slots"]:
                nlu["slots"]["nationality"] = []
            if isinstance(nlu["slots"]["nationality"], str):
                nlu["slots"]["nationality"] =  nlu["slots"]["nationality"].replace(" ","").split(",")
            if "category" not in nlu["slots"]:
                nlu["slots"]["category"] = []
            if isinstance(nlu["slots"]["category"], str):
                nlu["slots"]["category"] =  nlu["slots"]["category"].replace(" ","").split(",")
            if "ingredients" not in nlu["slots"]:
                nlu["slots"]["ingredients"] = []
            if isinstance(nlu["slots"]["ingredients"], str):
                nlu["slots"]["ingredients"] =  nlu["slots"]["ingredients"].replace(" ","").split(",")
            item["detected_slots"] = nlu["slots"]
    else:
        with open("test_data_recipe_reccomendation.json", "r") as f:
            test_data_recipe_reccomendation = json.load(f)
    # Calculate metrics
    metrics = calculate_nlu_metrics(test_data_recipe_reccomendation)
    #save metrics
    with open("nlu_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    # save test data recipe
    with open("test_data_recipe_reccomendation.json", "w") as f:
        json.dump(test_data_recipe_reccomendation, f, indent=4)
    print("Test data recipe reccomendation saved")
    pass