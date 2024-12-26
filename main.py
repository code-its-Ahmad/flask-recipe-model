import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngredientType(Enum):
 MAIN = "main"
 VEGETABLE = "vegetable"
 SPICE = "spice"
 LIQUID = "liquid"

@dataclass
class Ingredient:
 name: str
 amount: float
 unit: str
 type: IngredientType
 scaling_ratio: float # Ratio relative to main ingredient

# Enhanced sample recipe data with more structure
SAMPLE_RECIPE_DATA = {
 "vegetable_rice": {
 "base_recipe": {
 "ingredients": {
 "rice": Ingredient("rice", 0.5, "kg", IngredientType.MAIN, 1.0),
 "mixed_vegetables": Ingredient("mixed_vegetables", 0.2, "kg", IngredientType.VEGETABLE, 0.4),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.04)
 },
 "instructions": [
 "Wash and soak rice for 30 minutes",
 "Heat oil and sauté onions until golden",
 "Add vegetables and tomatoes, cook for 5 minutes",
 "Add spices and cook for 2 minutes",
 "Add rice and required water",
 "Cook until rice is done"
 ],
 "cooking_time": 45, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Adjust water based on rice type"
 }
 },
 "chicken_curry": {
 "base_recipe": {
 "ingredients": {
 "chicken": Ingredient("chicken", 0.5, "kg", IngredientType.MAIN, 1.0),
 "onion": Ingredient("onion", 0.2, "kg", IngredientType.VEGETABLE, 0.4),
 "tomato": Ingredient("tomato", 0.2, "kg", IngredientType.VEGETABLE, 0.4),
 "oil": Ingredient("oil", 0.1, "kg", IngredientType.LIQUID, 0.2),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.04),
 "ginger_garlic": Ingredient("ginger_garlic", 0.02, "kg", IngredientType.SPICE, 0.04)
 },
 "instructions": [
 "Marinate chicken with spices",
 "Heat oil and sauté onions",
 "Add ginger garlic and tomatoes",
 "Add chicken and cook until tender",
 "Garnish and serve"
 ],
 "cooking_time": 60,
 "difficulty": "medium",
 "serves": 4,
 "notes": "Marinate chicken for at least 30 minutes"
 }
 }
 ,
 "mutton_biryani": {
 "base_recipe": {
 "ingredients": {
 "mutton": Ingredient("mutton", 0.7, "kg", IngredientType.MAIN, 1.0),
 "rice": Ingredient("rice", 0.5, "kg", IngredientType.MAIN, 0.7),
 "onion": Ingredient("onion", 0.2, "kg", IngredientType.VEGETABLE, 0.3),
 "tomato": Ingredient("tomato", 0.15, "kg", IngredientType.VEGETABLE, 0.2),
 "yogurt": Ingredient("yogurt", 0.1, "kg", IngredientType.LIQUID, 0.15),
 "spices": Ingredient("spices", 0.03, "kg", IngredientType.SPICE, 0.05),
 "coriander": Ingredient("coriander", 0.05, "kg", IngredientType.VEGETABLE, 0.07)
 },
 "instructions": [
 "Marinate mutton with spices and yogurt for 1 hour",
 "Heat oil and sauté onions until golden",
 "Add tomatoes and cook until soft",
 "Add marinated mutton and cook until browned",
 "Add rice, water, and cook until rice is done"
 ],
 "cooking_time": 90, # in minutes
 "difficulty": "hard",
 "serves": 6,
 "notes": "Best served with raita"
 }
 }
}



SAMPLE_RECIPE_DATA.update({
 "paneer_tikka": {
 "base_recipe": {
 "ingredients": {
 "paneer": Ingredient("paneer", 0.3, "kg", IngredientType.MAIN, 1.0),
 "yogurt": Ingredient("yogurt", 0.1, "kg", IngredientType.LIQUID, 0.33),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.07),
 "lemon": Ingredient("lemon", 1, "piece", IngredientType.VEGETABLE, 0.05),
 "oil": Ingredient("oil", 0.03, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Cut paneer into cubes",
 "Mix yogurt, lemon juice, and spices to make the marinade",
 "Marinate paneer for 1 hour",
 "Skewer the marinated paneer and grill or bake until golden"
 ],
 "cooking_time": 30, # in minutes
 "difficulty": "medium",
 "serves": 4,
 "notes": "Serve with mint chutney"
 }
 },
 "dal_tadka": {
 "base_recipe": {
 "ingredients": {
 "lentils": Ingredient("lentils", 0.25, "kg", IngredientType.MAIN, 1.0),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.2),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.1)
 },
 "instructions": [
 "Cook lentils until soft",
 "In a separate pan, heat oil and sauté onions",
 "Add garlic and tomatoes, cook until tomatoes soften",
 "Add spices and cook for a minute",
 "Add cooked lentils to the pan and stir well",
 "Simmer for 10 minutes"
 ],
 "cooking_time": 40, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Serve with rice or roti"
 }
 },
 "pasta_alfredo": {
 "base_recipe": {
 "ingredients": {
 "pasta": Ingredient("pasta", 0.2, "kg", IngredientType.MAIN, 1.0),
 "butter": Ingredient("butter", 0.1, "kg", IngredientType.LIQUID, 0.5),
 "cream": Ingredient("cream", 0.2, "kg", IngredientType.LIQUID, 1.0),
 "parmesan_cheese": Ingredient("parmesan_cheese", 0.05, "kg", IngredientType.SPICE, 0.25),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "black_pepper": Ingredient("black_pepper", 0.01, "kg", IngredientType.SPICE, 0.05)
 },
 "instructions": [
 "Boil pasta until al dente",
 "In a pan, melt butter and sauté garlic",
 "Add cream and bring to a simmer",
 "Add parmesan cheese and stir until sauce thickens",
 "Mix in cooked pasta and coat well with sauce",
 "Serve with black pepper on top"
 ],
 "cooking_time": 25, # in minutes
 "difficulty": "medium",
 "serves": 2,
 "notes": "Best served with garlic bread"
 }
 },
 "beef_stew": {
 "base_recipe": {
 "ingredients": {
 "beef": Ingredient("beef", 0.6, "kg", IngredientType.MAIN, 1.0),
 "potatoes": Ingredient("potatoes", 0.4, "kg", IngredientType.VEGETABLE, 0.3),
 "carrots": Ingredient("carrots", 0.2, "kg", IngredientType.VEGETABLE, 0.3),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.05)
 },
 "instructions": [
 "Brown the beef chunks in oil",
 "Add onions and sauté until soft",
 "Add carrots, potatoes, and water, bring to a boil",
 "Add spices and simmer for 1 hour until beef is tender"
 ],
 "cooking_time": 90, # in minutes
 "difficulty": "medium",
 "serves": 6,
 "notes": "Serve with bread or rice"
 }
 },
 "fish_fry": {
 "base_recipe": {
 "ingredients": {
 "fish": Ingredient("fish", 0.5, "kg", IngredientType.MAIN, 1.0),
 "turmeric": Ingredient("turmeric", 0.01, "kg", IngredientType.SPICE, 0.02),
 "chili_powder": Ingredient("chili_powder", 0.02, "kg", IngredientType.SPICE, 0.04),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.04),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Marinate fish with turmeric, chili powder, and garlic",
 "Heat oil in a pan",
 "Fry fish until golden and crispy on both sides",
 "Serve with lemon wedges and onion rings"
 ],
 "cooking_time": 30, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Can be served as a snack or with rice"
 }
 },
 "paneer_tikka": {
 "base_recipe": {
 "ingredients": {
 "paneer": Ingredient("paneer", 0.3, "kg", IngredientType.MAIN, 1.0),
 "yogurt": Ingredient("yogurt", 0.1, "kg", IngredientType.LIQUID, 0.33),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.07),
 "lemon": Ingredient("lemon", 1, "piece", IngredientType.VEGETABLE, 0.05),
 "oil": Ingredient("oil", 0.03, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Cut paneer into cubes",
 "Mix yogurt, lemon juice, and spices to make the marinade",
 "Marinate paneer for 1 hour",
 "Skewer the marinated paneer and grill or bake until golden"
 ],
 "cooking_time": 30, # in minutes
 "difficulty": "medium",
 "serves": 4,
 "notes": "Serve with mint chutney"
 }
 },
 "dal_tadka": {
 "base_recipe": {
 "ingredients": {
 "lentils": Ingredient("lentils", 0.25, "kg", IngredientType.MAIN, 1.0),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.2),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.1)
 },
 "instructions": [
 "Cook lentils until soft",
 "In a separate pan, heat oil and sauté onions",
 "Add garlic and tomatoes, cook until tomatoes soften",
 "Add spices and cook for a minute",
 "Add cooked lentils to the pan and stir well",
 "Simmer for 10 minutes"
 ],
 "cooking_time": 40, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Serve with rice or roti"
 }
 },
 "pasta_alfredo": {
 "base_recipe": {
 "ingredients": {
 "pasta": Ingredient("pasta", 0.2, "kg", IngredientType.MAIN, 1.0),
 "butter": Ingredient("butter", 0.1, "kg", IngredientType.LIQUID, 0.5),
 "cream": Ingredient("cream", 0.2, "kg", IngredientType.LIQUID, 1.0),
 "parmesan_cheese": Ingredient("parmesan_cheese", 0.05, "kg", IngredientType.SPICE, 0.25),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "black_pepper": Ingredient("black_pepper", 0.01, "kg", IngredientType.SPICE, 0.05)
 },
 "instructions": [
 "Boil pasta until al dente",
 "In a pan, melt butter and sauté garlic",
 "Add cream and bring to a simmer",
 "Add parmesan cheese and stir until sauce thickens",
 "Mix in cooked pasta and coat well with sauce",
 "Serve with black pepper on top"
 ],
 "cooking_time": 25, # in minutes
 "difficulty": "medium",
 "serves": 2,
 "notes": "Best served with garlic bread"
 }
 },
 "beef_stew": {
 "base_recipe": {
 "ingredients": {
 "beef": Ingredient("beef", 0.6, "kg", IngredientType.MAIN, 1.0),
 "potatoes": Ingredient("potatoes", 0.4, "kg", IngredientType.VEGETABLE, 0.3),
 "carrots": Ingredient("carrots", 0.2, "kg", IngredientType.VEGETABLE, 0.3),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.05)
 },
 "instructions": [
 "Brown the beef chunks in oil",
 "Add onions and sauté until soft",
 "Add carrots, potatoes, and water, bring to a boil",
 "Add spices and simmer for 1 hour until beef is tender"
 ],
 "cooking_time": 90, # in minutes
 "difficulty": "medium",
 "serves": 6,
 "notes": "Serve with bread or rice"
 }
 },
 "fish_fry": {
 "base_recipe": {
 "ingredients": {
 "fish": Ingredient("fish", 0.5, "kg", IngredientType.MAIN, 1.0),
 "turmeric": Ingredient("turmeric", 0.01, "kg", IngredientType.SPICE, 0.02),
 "chili_powder": Ingredient("chili_powder", 0.02, "kg", IngredientType.SPICE, 0.04),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.04),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Marinate fish with turmeric, chili powder, and garlic",
 "Heat oil in a pan",
 "Fry fish until golden and crispy on both sides",
 "Serve with lemon wedges and onion rings"
 ],
 "cooking_time": 30, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Can be served as a snack or with rice"
 }
 },
 "veg_biryani": {
 "base_recipe": {
 "ingredients": {
 "rice": Ingredient("rice", 0.5, "kg", IngredientType.MAIN, 1.0),
 "mixed_vegetables": Ingredient("mixed_vegetables", 0.3, "kg", IngredientType.VEGETABLE, 0.6),
 "onion": Ingredient("onion", 0.15, "kg", IngredientType.VEGETABLE, 0.3),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "spices": Ingredient("spices", 0.03, "kg", IngredientType.SPICE, 0.1),
 "yogurt": Ingredient("yogurt", 0.1, "kg", IngredientType.LIQUID, 0.3),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Soak rice for 30 minutes",
 "Heat oil in a pan and sauté onions and garlic until golden",
 "Add tomatoes and spices, cook for 5 minutes",
 "Add vegetables and cook for another 5 minutes",
 "Add yogurt and mix well",
 "Add rice, water, and cook until rice is tender"
 ],
 "cooking_time": 60, # in minutes
 "difficulty": "medium",
 "serves": 4,
 "notes": "Serve with raita"
 }
 },
 "chole_bhature": {
 "base_recipe": {
 "ingredients": {
 "chickpeas": Ingredient("chickpeas", 0.25, "kg", IngredientType.MAIN, 1.0),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.08),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1),
 "flour": Ingredient("flour", 0.25, "kg", IngredientType.MAIN, 1.0),
 "yogurt": Ingredient("yogurt", 0.1, "kg", IngredientType.LIQUID, 0.4)
 },
 "instructions": [
 "Soak chickpeas overnight and cook until soft",
 "Heat oil in a pan and sauté onions, tomatoes, and spices",
 "Add cooked chickpeas and simmer for 20 minutes",
 "Make dough for bhature using flour, yogurt, and water",
 "Roll the dough into small balls and deep fry until golden"
 ],
 "cooking_time": 120, # in minutes (soaking time included)
 "difficulty": "high",
 "serves": 4,
 "notes": "Serve hot with pickles and onion salad"
 }
 },
 "vegetable_pulao": {
 "base_recipe": {
 "ingredients": {
 "rice": Ingredient("rice", 0.5, "kg", IngredientType.MAIN, 1.0),
 "mixed_vegetables": Ingredient("mixed_vegetables", 0.3, "kg", IngredientType.VEGETABLE, 0.6),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.05),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Heat oil in a pan and sauté onions and garlic",
 "Add vegetables and cook for 5 minutes",
 "Add spices and cook for 2 minutes",
 "Add soaked rice and water, cook until rice is done"
 ],
 "cooking_time": 40, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Serve with yogurt or raita"
 }
 },
 "butter_chicken": {
 "base_recipe": {
 "ingredients": {
 "chicken": Ingredient("chicken", 0.6, "kg", IngredientType.MAIN, 1.0),
 "butter": Ingredient("butter", 0.1, "kg", IngredientType.LIQUID, 0.2),
 "cream": Ingredient("cream", 0.1, "kg", IngredientType.LIQUID, 0.2),
 "tomato": Ingredient("tomato", 0.2, "kg", IngredientType.VEGETABLE, 0.4),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.05)
 },
 "instructions": [
 "Marinate chicken with yogurt and spices for 30 minutes",
 "Cook chicken in butter until golden",
 "In a separate pan, sauté onions and tomatoes",
 "Add cream and spices, cook for 5 minutes",
 "Add cooked chicken to the sauce and simmer for 10 minutes"
 ],
 "cooking_time": 60, # in minutes
 "difficulty": "medium",
 "serves": 4,
 "notes": "Serve with naan or rice"
 }
 },
 "masoor_dal": {
 "base_recipe": {
 "ingredients": {
 "red_lentils": Ingredient("red_lentils", 0.25, "kg", IngredientType.MAIN, 1.0),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "tomato": Ingredient("tomato", 0.1, "kg", IngredientType.VEGETABLE, 0.4),
 "garlic": Ingredient("garlic", 0.02, "kg", IngredientType.SPICE, 0.1),
 "spices": Ingredient("spices", 0.02, "kg", IngredientType.SPICE, 0.05),
 "oil": Ingredient("oil", 0.05, "kg", IngredientType.LIQUID, 0.1)
 },
 "instructions": [
 "Cook red lentils until soft",
 "In a separate pan, heat oil and sauté onions, tomatoes, and garlic",
 "Add spices and cook for a minute",
 "Add cooked lentils to the pan, stir, and simmer for 10 minutes"
 ],
 "cooking_time": 40, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Serve with rice or bread"
 }
 },
 "vegetable_soup": {
 "base_recipe": {
 "ingredients": {
 "carrots": Ingredient("carrots", 0.2, "kg", IngredientType.VEGETABLE, 0.4),
 "onion": Ingredient("onion", 0.1, "kg", IngredientType.VEGETABLE, 0.2),
 "potatoes": Ingredient("potatoes", 0.3, "kg", IngredientType.VEGETABLE, 0.6),
 "spices": Ingredient("spices", 0.01, "kg", IngredientType.SPICE, 0.05),
 "water": Ingredient("water", 1.0, "L", IngredientType.LIQUID, 1.0)
 },
 "instructions": [
 "Chop vegetables into small pieces",
 "Heat oil in a pot and sauté onions",
 "Add carrots, potatoes, and spices",
 "Add water and bring to a boil",
 "Simmer for 30 minutes until vegetables are tender"
 ],
 "cooking_time": 45, # in minutes
 "difficulty": "easy",
 "serves": 4,
 "notes": "Serve hot with a side of bread"
 }
 }
})

@dataclass
class RecipeResult:
 recipe_name: str
 input_quantities: Dict[str, Tuple[float, str]]
 scaled_recipe: Dict
 recommended_recipe: Optional[Dict] = None
 success: bool = True
 error_message: Optional[str] = None
 cooking_time: int = 0
 serves: int = 0
 notes: List[str] = None

class ImprovedRecipeTester:
    def __init__(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_quantities(self, test_input: str) -> Dict[str, Tuple[float, str]]:
        """Enhanced quantity extraction with better pattern matching"""
        quantities = {}
        patterns = {
            'kg': r'(\d+(?:\.\d+)?)\s*kg\s+(?:of\s+)?(\w+(?:\s+\w+)*)',
            'g': r'(\d+(?:\.\d+)?)\s*g(?:rams)?\s+(?:of\s+)?(\w+(?:\s+\w+)*)',
            'ml': r'(\d+(?:\.\d+)?)\s*ml\s+(?:of\s+)?(\w+(?:\s+\w+)*)',
        }

        for unit, pattern in patterns.items():
            matches = re.finditer(pattern, test_input.lower(), re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1))
                ingredient = match.group(2).strip().replace(" ", "_")
                # Convert to standard units (kg)
                if unit == 'g':
                    amount = amount / 1000
                    unit = 'kg'
                elif unit == 'ml':
                    amount = amount / 1000
                    unit = 'kg'

                quantities[ingredient] = (amount, unit)

        return quantities

    def calculate_recipe_scaling(self,
                                 base_recipe: Dict,
                                 requested_quantities: Dict[str, Tuple[float, str]]) -> Tuple[Dict, Dict]:
        """Calculate both user-requested and recommended recipe quantities"""
        scaled_recipe = {'ingredients': {}, 'instructions': base_recipe['instructions'].copy()}
        recommended_recipe = {'ingredients': {}, 'instructions': base_recipe['instructions'].copy()}

        # Find main ingredient and its scaling factor
        main_ingredient = next(ing for ing, details in base_recipe['ingredients'].items()
                               if details.type == IngredientType.MAIN)
        base_amount = base_recipe['ingredients'][main_ingredient].amount
        requested_amount = requested_quantities.get(main_ingredient, (base_amount, 'kg'))[0]
        scaling_factor = requested_amount / base_amount

        # Scale ingredients for both recipes
        for ing_name, ing_details in base_recipe['ingredients'].items():
            # User requested recipe
            if ing_name in requested_quantities:
                scaled_recipe['ingredients'][ing_name] = requested_quantities[ing_name]
            else:
                scaled_amount = ing_details.amount * scaling_factor
                scaled_recipe['ingredients'][ing_name] = (scaled_amount, ing_details.unit)

            # Recommended recipe
            recommended_amount = ing_details.amount * scaling_factor
            recommended_recipe['ingredients'][ing_name] = (recommended_amount, ing_details.unit)

        return scaled_recipe, recommended_recipe

    def adjust_instructions(self, instructions: List[str], scaling_factor: float) -> List[str]:
        """Adjust cooking instructions based on quantity scaling"""
        adjusted = []
        for instruction in instructions:
            if any(word in instruction.lower() for word in ['cook', 'sauté', 'simmer', 'boil']):
                time_adjustment = f" (cooking time adjusted to {int(scaling_factor * 100)}% for scaled quantity)"
                adjusted.append(instruction + time_adjustment)
            else:
                adjusted.append(instruction)
        return adjusted

    def run_test(self, test_input: str) -> RecipeResult:
        """Enhanced test execution with better error handling and recommendations"""
        try:
            # Extract quantities
            quantities = self.extract_quantities(test_input)

            # Find recipe name
            recipe_name = None
            for name in SAMPLE_RECIPE_DATA.keys():
                if name.replace('_', ' ') in test_input.lower():
                    recipe_name = name
                    break

            if not recipe_name:
                raise ValueError("Recipe not found in test input")

            # Get base recipe
            base_recipe = SAMPLE_RECIPE_DATA[recipe_name]['base_recipe']

            # Scale recipe
            scaled_recipe, recommended_recipe = self.calculate_recipe_scaling(base_recipe, quantities)

            # Calculate serves
            base_serves = base_recipe['serves']
            main_ingredient = next(ing for ing, details in base_recipe['ingredients'].items()
                                   if details.type == IngredientType.MAIN)
            scaling_factor = scaled_recipe['ingredients'][main_ingredient][0] / base_recipe['ingredients'][main_ingredient].amount
            scaled_serves = int(base_serves * scaling_factor)

            # Adjust cooking time
            scaled_time = int(base_recipe['cooking_time'] * (1 + (scaling_factor - 1) * 0.5))

            # Generate notes
            notes = [base_recipe['notes']]
            if scaling_factor > 2:
                notes.append("Consider cooking in multiple batches for better results")
            if scaling_factor < 0.5:
                notes.append("Reduce cooking time to avoid overcooking")

            return RecipeResult(
                recipe_name=recipe_name,
                input_quantities=quantities,
                scaled_recipe=scaled_recipe,
                recommended_recipe=recommended_recipe,
                success=True,
                cooking_time=scaled_time,
                serves=scaled_serves,
                notes=notes
            )

        except Exception as e:
            logger.error(f"Error processing recipe: {str(e)}")
            return RecipeResult(
                recipe_name="unknown",
                input_quantities={},
                scaled_recipe={},
                success=False,
                error_message=str(e)
            )


def print_recipe_result(result: RecipeResult):
    """Enhanced result printing with more details"""
    print("\n" + "="*50)
    print(f"Recipe: {result.recipe_name}")
    print("="*50)

    if result.success:
        print("\nYOUR REQUESTED RECIPE:")
        print("-" * 20)
        print("\nIngredients:")
        for ing, (amount, unit) in result.scaled_recipe['ingredients'].items():
            print(f"- {ing}: {amount:.3f}{unit}")

        if result.recommended_recipe:
            print("\nRECOMMENDED PROPORTIONS:")
            print("-" * 20)
            print("\nIngredients:")
            for ing, (amount, unit) in result.recommended_recipe['ingredients'].items():
                print(f"- {ing}: {amount:.3f}{unit}")

        print("\nInstructions:")
        for i, instruction in enumerate(result.scaled_recipe['instructions'], 1):
            print(f"{i}. {instruction}")

        print(f"\nCooking Time: {result.cooking_time} minutes")
        print(f"Serves: {result.serves} people")

        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")
    else:
        print(f"Test failed: {result.error_message}")


def run_tests():
    """Run test cases"""
    tester = ImprovedRecipeTester()

    test_cases = [
        "I want to make vegetable rice with 10kg rice and 1kg vegetables",
        "Make chicken curry with 2kg chicken",
        "I need vegetable rice recipe for 5kg rice",
    ]

    logger.info("Starting recipe tests...")

    for test_input in test_cases:
        logger.info(f"Testing input: {test_input}")
        result = tester.run_test(test_input)
        print_recipe_result(result)

if __name__ == "__main__":
    run_tests()
