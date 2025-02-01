
import json
import random
from data.database import get_all_areas, get_all_categories, get_all_ingredients, get_all_recipe_names
from pipeline import generate_dm_input, generate_dm_output, get_args
from recipe_state_tracker import RecipeStateTracker
from utils import load_model
from sklearn.metrics import precision_recall_fscore_support
import time

def compute_metrics(ground_truth_list, prediction_list):
    """Compute precision, recall, and F1-score for a single pair of lists."""
    gt_set = set(ground_truth_list)
    pred_set = set(prediction_list)

    TP = len(gt_set & pred_set)  # True Positives (Correct predictions)
    FP = len(pred_set - gt_set)  # False Positives (Wrongly predicted)
    FN = len(gt_set - pred_set)  # False Negatives (Missed predictions)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":

    args = get_args()
    model, tokenizer = load_model(args)
    start_time = time.time()
    test_data = []
    predictions = []

    for intent in ["ask_for_ingredients", "ask_for_procedure", "ask_for_time"]:
        for _ in range(20):
            state_tracker = RecipeStateTracker()

            if intent == "recipe_recommendation":
                all_category = get_all_categories()
                all_nationality = get_all_areas()
                all_ingredients = get_all_ingredients()
                num_filters = random.choices([1, 2, 3], weights=[0.6, 0.2, 0.2], k=1)[0]
                filters = ["category", "ingredients", "nationality"]
                selected_filters = random.sample(filters, num_filters)

                category = random.choice(all_category) if "category" in selected_filters else None
                ingredients = [random.choice(all_ingredients) for _ in range(random.choice(range(1, 3)))] if "ingredients" in selected_filters else None
                all_nationality = random.choice(all_nationality) if "nationality" in selected_filters else None
                nlu = {
                    "intent": "recipe_recommendation",
                    "slots": {
                        "category": category,
                        "ingredients": ingredients,
                        "nationality": all_nationality,
                    }
                }
            else:
                all_recipes = get_all_recipe_names()
                recipe = random.choice(all_recipes) if random.random() > 0.2 else None
                nlu = {
                    "intent": intent,
                    "slots": {
                        "recipe_name": recipe,
                    }
                }

            state_tracker.update(nlu)
            dm_input, filtered_recipes, recipe_information = generate_dm_input(nlu, state_tracker)

            dm_output = generate_dm_output(nlu, dm_input, filtered_recipes, recipe_information, model, tokenizer, args, False, True)

            if intent == "recipe_recommendation":
                actions = []
                if len(filtered_recipes) > 0:
                    actions.append("propose_recipe")
                    for key in nlu["slots"]:
                        if nlu["slots"][key] is None:
                            actions.append(f"req_info_{key}")
                else:
                    actions.append("no_recipe_found")

                data = {
                    "nlu": nlu,
                    "dm_input": dm_input,
                    "dm_output": dm_output,
                    "actions": actions
                }
            else:
                actions = []

                if intent == "ask_for_ingredients":
                    if recipe_information is None:
                        actions.append("ask_recipe_name")
                    else:
                        actions.append("provide_ingredients")
                elif intent == "ask_for_procedure":
                    if recipe_information is None:
                        actions.append("ask_recipe_name")
                    else:
                        actions.append("provide_procedure")

                elif intent == "ask_for_time":
                    if recipe_information is None:
                        actions.append("ask_recipe_name")
                    else:
                        actions.append("provide_time_needed")

                elif intent == "not_supported":
                    actions.append("tell to the user that the bot cannot help for his request or it has understood wrong, ask to the user to repeat his intention.")

                data = {
                    "nlu": nlu,
                    "dm_input": dm_input,
                    "dm_output": dm_output,
                    "actions": actions
                }
            test_data.append(data)
            predictions.append((dm_output["action_required"], actions))

    with open("data/test_data.json", "w") as f:
        json.dump(test_data, f, indent=4)

    results = [compute_metrics(gt, pred) for gt, pred in predictions]

    # Aggregate results (micro-average over all pairs)
    precision_avg = sum(p for p, r, f in results) / len(results)
    recall_avg = sum(r for p, r, f in results) / len(results)
    f1_avg = sum(f for p, r, f in results) / len(results)

    metrics = {
        "precision": precision_avg,
        "recall": recall_avg,
        "f1": f1_avg
    }

    with open("data/dm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script duration: {duration} seconds")