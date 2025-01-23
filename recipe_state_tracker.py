import json
from data.database import get_all_categories, get_all_ingredients, get_all_areas
from rule import *

class Intent:
    def __init__(self):
        self.active = False
        self.intent = None
        self.slots = {}
        self.values_allowed_slots: dict[str,Rule] = {}

    def get_active(self):
        return self.active
    
    def set_slot(self, intent, slot_key, slot_value):
        self.slots[intent][slot_key] = slot_value
    
    def get_slot(self, intent, slot_key):
        return self.slots[intent][slot_key]

    def get_slot_value(self, slot_key):
        return self.slots[slot_key]

    def set_slots(self, intent, slots):
        self.slots[intent] = slots
    
    def get_slots(self, intent):
        return self.slots[intent]

    def get_available_slots(self):
        return self.values_allowed_slots

    def get_slots(self):
        return self.slots
    
    def get_intent(self):
        return self.intent
    
    def to_dict(self):
        return {
            "intent": self.intent,
            "slots": self.slots
        }

class RecipeStateTracker:
    def __init__(self):
        self.selected_recipe = None
        self.intents: dict[str, Intent] = {
            "recipe_recommendation": RecipeRaccomandation(),
            "ask_for_ingredients": AskForIngredients(),
            "ask_for_procedure": AskForProcedure(),
            "ask_for_time": AskForTime(),
            "not_supported": Intent(),
            "end_conversation": Intent(),
        }   
        
        self.slots = {}
        
        for intent in self.intents.keys():
            self.slots[intent] = self.intents[intent].get_slots()

        self.values_allowed_slots = {}

        for intent in self.intents.keys():
            self.values_allowed_slots[intent] = self.intents[intent].get_available_slots()

    def update(self, nlu_data):
        if 'intent' in nlu_data.keys():
            valid_intent = self.__update_intent(nlu_data['intent'])
            if valid_intent:
                if 'slots' in nlu_data.keys():
                    self.__update_slots(nlu_data['slots'], nlu_data['intent'])

    def __update_intent(self, intent :str) -> bool:
        if intent in self.intents.keys():
            self.intents[intent].active = True
            return True
        else:
            print(f"Invalid intent: {intent}")
            return False

    def __update_slots(self, slots : dict, intent :str):
        for slot, value in slots.items():
            if slot in self.intents[intent].get_slots().keys() and value:
                if slot == "ingredients" and value:
                    if isinstance(value, str):
                        list_value = value.lower().split(",")
                    elif isinstance(value, list):
                        list_value = value
                    self.intents[intent].slots[slot] = []
                    for ing in list_value:
                        ing = ing.strip().lower()
                        if self.intents[intent].values_allowed_slots[slot].validate(ing):
                            self.intents[intent].slots[slot].append(ing)
                        else:
                            print(f"Invalid ingredient: {ing}")
                else:    
                    if isinstance(value, str):
                        value = value.lower()
                    if self.intents[intent].values_allowed_slots[slot].validate(value): 
                        self.intents[intent].slots[slot] = value
                    else:
                        print(f"Invalid value for slot {slot}: {value}")
            else:
                pass

    def to_dict(self):
        state_dict = {}
        for intent in self.intents.keys():
            if self.intents[intent].active:
                state_dict[intent] = self.intents[intent].to_dict()
        return state_dict

    def get_slots(self, intent):
        return self.slots[intent]

    def to_string(self):
        return json.dumps(self.to_dict(), indent=4)
    
class AskForTime(Intent):
    def __init__(self):
        super().__init__()
        self.intent = "ask_for_time"
        self.slots = {
            "recipe_name": None,
        }
        self.values_allowed_slots = {
            "recipe_name": IsStringRule(),
        }
    
class RecipeRaccomandation(Intent):
    def __init__(self):
        super().__init__()
        self.intent = "recipe_recommendation"
        self.slots = {
            "nationality": None,
            "category": None,
            "ingredients": None,
        }

        all_ingredients = get_all_ingredients()
        all_ingredients = [ing.lower() for ing in all_ingredients if ing]

        all_areas = get_all_areas()
        all_areas = [area.lower() for area in all_areas if area]

        all_categories = get_all_categories()
        all_categories = [category.lower() for category in all_categories]

        self.values_allowed_slots = {
            "nationality": InListRule(all_areas),
            "ingredients": InListRule(all_ingredients),
            "category": InListRule(all_categories),
        }

class AskForIngredients(Intent):
    def __init__(self):
        super().__init__()
        self.intent = "ask_for_ingredients"
        self.slots = {
            "recipe_name": None,
        }
        self.values_allowed_slots = {
            "recipe_name": IsStringRule(),
        }

class NotSupported(Intent):
    def __init__(self):
        super().__init__()
        self.intent = "not_supported"
        self.slots = {}
        self.values_allowed_slots = {}

class AskForProcedure(Intent):
    def __init__(self):
        super().__init__()
        self.intent = "ask_for_procedure"
        self.slots = {
            "recipe_name": None,
        }
        self.values_allowed_slots = {
            "recipe_name": IsStringRule(),
        }

