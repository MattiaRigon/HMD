import os
import json
from pathlib import Path
from typing import Optional, List
import requests
from data.config import API_URL

# Define the structure for Meal class
class Meal:
    def __init__(
        self,
        idMeal: str,
        strMeal: str,
        strCategory: Optional[str] = None,
        strArea: Optional[str] = None,
        strInstructions: Optional[str] = None,
        strMealThumb: Optional[str] = None,
        strTags: Optional[str] = None,
        strYoutube: Optional[str] = None,
        strSource: Optional[str] = None,
        ingredients: Optional[str] = None,
    ):
        self.idMeal = idMeal
        self.strMeal = strMeal
        self.strCategory = strCategory
        self.strArea = strArea
        self.strInstructions = strInstructions
        self.strMealThumb = strMealThumb
        self.strTags = strTags
        self.strYoutube = strYoutube
        self.strSource = strSource
        self.ingredients = ingredients

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data):
        return Meal(**data)

# Path to the JSON database
DB_PATH = "data/meal_database.json"

# Initialize the database
def init_db():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w") as db_file:
            json.dump([], db_file)

# Fetch data from the MealDB API
def fetch_meals():
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()

    meals = []
    valid_fields = Meal.__init__.__code__.co_varnames

    for item in data.get("meals", []):
        # Prepare ingredients field
        item["ingredients"] = [item[f"strIngredient{i}"] for i in range(1, 21) if item[f"strIngredient{i}"]]
        item["ingredients"] = "##".join(item["ingredients"])

        # Filter out unexpected fields
        filtered_item = {key: value for key, value in item.items() if key in valid_fields}

        # Create a Meal instance
        try:
            meals.append(Meal.from_dict(filtered_item))
        except TypeError as e:
            print(f"Error creating Meal: {e}")

        if len(meals) >= 100:  # Limit to 10 meals for demonstration
            break

    return meals


# Insert meals into the JSON database
def insert_meals(meals: List[Meal]):
    with open(DB_PATH, "r") as db_file:
        db_data = json.load(db_file)

    db_data_ids = {meal["idMeal"] for meal in db_data}

    for meal in meals:
        if meal.idMeal not in db_data_ids:
            db_data.append(meal.to_dict())

    with open(DB_PATH, "w") as db_file:
        json.dump(db_data, db_file, indent=4)

# Query functions
def get_all_meals():
    with open(DB_PATH, "r") as db_file:
        db_data = json.load(db_file)
        return [Meal.from_dict(meal) for meal in db_data]

def get_meal_by_name(name: str):
    return [meal.to_dict() for meal in get_all_meals() if name.lower() in meal.strMeal.lower()]

def get_meals_by_category(category: str):
    return [meal.to_dict() for meal in get_all_meals() if meal.strCategory == category]

def get_all_ingredients():
    ingredients = []
    for meal in get_all_meals():
        ingredients.extend(meal.ingredients.split("##"))
    return list(set(ingredients))

def get_ingredients_by_meal(name: str):
    meals = get_meal_by_name(name)
    ingredients = []
    for meal in meals:
        ingredients.extend(meal.ingredients.split("##"))
    return list(set(ingredients))

def filter_recipes(nationality: Optional[str] = None, category: Optional[str] = None, ingredients: Optional[List[str]] = None):
    results = []
    if isinstance(ingredients,str):
        ingredients = ingredients.split(",")
        ingredients = [ing.replace(" ","").lower() for ing in ingredients]
    for meal in get_all_meals():
        if (nationality is None or meal.strArea.lower() == nationality.lower()) and \
            (category is None or meal.strCategory.lower() == category.lower()) and \
            (ingredients is None or all(ingredient in meal.ingredients.lower().split("##") for ingredient in ingredients)):
            results.append(meal.strMeal)
    return results

def get_all_areas():
    return list(set(meal.strArea for meal in get_all_meals() if meal.strArea))

def get_recipes(slots: dict):
    results = []
    for meal in get_all_meals():
        if all(getattr(meal, key) == value for key, value in slots.items() if value):
            results.append(meal.strMeal)
    return results

def get_meals_by_ingredients(ingredients: List[str]):
    results = []
    for meal in get_all_meals():
        meal_ingredients = set(meal.ingredients.split("##"))
        if all(ingredient in meal_ingredients for ingredient in ingredients):
            results.append(meal.strMeal)
    return results

def get_all_categories():
    return list(set(meal.strCategory for meal in get_all_meals() if meal.strCategory))

if __name__ == "__main__":
    # Initialize the database
    init_db()

    # Fetch and insert meals
    meals = fetch_meals()
    if meals:
        print(f"Fetched {len(meals)} meals from the API.")
        insert_meals(meals)
    else:
        print("No meals found in the API.")

    # Example queries
    print("All meals:", [meal.strMeal for meal in get_all_meals()])
    print("Meals with 'Lasagne':", [meal.strMeal for meal in get_meal_by_name("Lasagne")])
    print("Meals in 'Seafood' category:", [meal.strMeal for meal in get_meals_by_category("Seafood")])
    print("All ingredients:", get_all_ingredients())
