import os
from pathlib import Path
from sqlmodel import Field, SQLModel, Session, select
from typing import Optional, List
import requests
from data.config import API_URL, engine

# Define the SQLModel for Meal
class Meal(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)
    idMeal: str = Field(index=True, unique=True)
    strMeal: str
    strCategory: Optional[str] = None
    strArea: Optional[str] = None
    strInstructions: Optional[str] = None
    strMealThumb: Optional[str] = None
    strTags: Optional[str] = None
    strYoutube: Optional[str] = None
    strSource: Optional[str] = None
    ingredients: Optional[str] = None


# Initialize the database
def init_db():
    SQLModel.metadata.create_all(engine)


# Fetch data from the MealDB API
def fetch_meals():
    response = requests.get(API_URL)
    response.raise_for_status()  # Raise an error for bad HTTP responses
    data = response.json()

    meals = []

    for item in data.get("meals", []):
        item["ingredients"] = [item[f"strIngredient{i}"] for i in range(1, 21) if item[f"strIngredient{i}"]]
        item["ingredients"] = "##".join(item["ingredients"])
        meals.append(Meal(**item))
        if len(meals) >= 10:
            break
    return meals


# Insert meals into the database
def insert_meals(meals: List[Meal]):
    with Session(engine) as session:
        for meal in meals:
            existing = session.exec(select(Meal).where(Meal.idMeal == meal.idMeal)).first()
            if not existing:  # Avoid duplicates
                session.add(meal)
        session.commit()


# Query functions
def get_meal_by_name(name: str):
    with Session(engine) as session:
        statement = select(Meal).where(Meal.strMeal.contains(name))
        results = session.exec(statement).all()
        return results


def get_meals_by_category(category: str):
    with Session(engine) as session:
        statement = select(Meal).where(Meal.strCategory == category)
        results = session.exec(statement).all()
        return results


def get_all_meals():
    with Session(engine) as session:
        statement = select(Meal)
        results = session.exec(statement).all()
        return results
    
def get_all_ingridients():

    with Session(engine) as session:
        statement = select(Meal)
        results = session.exec(statement).all()
        ingridients = []
        for meal in results:
            ingridients.extend(meal.ingredients.split("##"))
        return ingridients

def get_ingridients_by_meal(name: str):
    with Session(engine) as session:
        statement = select(Meal).where(Meal.strMeal.contains(name))
        results = session.exec(statement).all()
        ingridients = []
        for meal in results:
            ingridients.extend(meal.ingredients.split("##"))
        return list(set(ingridients))
    
def get_all_area():
    with Session(engine) as session:
        statement = select(Meal)
        results = session.exec(statement).all()
        areas = []
        for meal in results:
            if meal.strArea not in areas:
                areas.append(meal.strArea)
        return list(set(areas))
    
def get_meals_by_ingridients(ingridients: str):
    with Session(engine) as session:
        statement = select(Meal)
        results = session.exec(statement).all()
        final_results = []
        for meal in results:
            count = True
            for ingridient in ingridients:
                if ingridient in meal.ingredients:
                    continue
                else:
                    count = False
            if count:
                final_results.append(meal.strMeal)
        return final_results

if __name__ == "__main__":
    # Initialize the database
    # init_db()

    # # Fetch and insert meals
    # meals = fetch_meals()
    # if meals:
    #     print(f"Fetched {len(meals)} meals from the API.")
    #     insert_meals(meals)
    # else:
    #     print("No meals found in the API.")

    # Example queries
    # print("All meals:", get_all_meals())
    # meals = get_all_meals()
    # for meal in meals:
    #     print(meal.strMeal)
        
    # print("Meals with 'Lasagne':", get_meal_by_name("Lasagne"))
    # print("Meals in 'Seafood' category:", get_meals_by_category("Seafood"))
    # print("All meals:", get_all_meals())
    print("All ingridients:", get_all_ingridients())
