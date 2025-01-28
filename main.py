import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Tuple, Optional
import re
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enum for Ingredient Types
class IngredientType(Enum):
    MAIN = "main"
    VEGETABLE = "vegetable"
    SPICE = "spice"
    LIQUID = "liquid"

# Dataclass for Ingredient
@dataclass
class Ingredient:
    name: str
    amount: float
    unit: str
    type: IngredientType
    scaling_ratio: float

# Load and parse the recipe data
def load_recipe_data(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        for recipe_name, recipe_data in data.items():
            for ingredient_name, ingredient_data in recipe_data['base_recipe']['ingredients'].items():
                recipe_data['base_recipe']['ingredients'][ingredient_name] = Ingredient(
                    name=ingredient_data['name'],
                    amount=ingredient_data['amount'],
                    unit=ingredient_data['unit'],
                    type=IngredientType(ingredient_data['type']),
                    scaling_ratio=ingredient_data['scaling_ratio']
                )
        logger.info("Recipe data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading recipe data: {str(e)}")
        raise

# Dataclass for Recipe Result
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

# Recipe Tester Class
class ImprovedRecipeTester:
    def __init__(self, recipe_data: Dict):
        self.recipe_data = recipe_data
        try:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_quantities(self, test_input: str) -> Dict[str, Tuple[float, str]]:
        """Extract ingredient quantities from user input"""
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

                # Convert to standard units
                if unit == 'g':
                    amount /= 1000  # Convert grams to kg
                    unit = 'kg'
                elif unit == 'ml':
                    amount /= 1000  # Convert ml to liters (treated as kg equivalent)

                quantities[ingredient] = (amount, unit)

        return quantities

    def calculate_recipe_scaling(self, base_recipe: Dict, requested_quantities: Dict[str, Tuple[float, str]]) -> Tuple[Dict, Dict]:
        """Calculate scaled and recommended recipe quantities"""
        scaled_recipe = {'ingredients': {}, 'instructions': base_recipe['instructions'].copy()}
        recommended_recipe = {'ingredients': {}, 'instructions': base_recipe['instructions'].copy()}

        # Determine scaling factor based on main ingredient
        main_ingredient = next(ing for ing, details in base_recipe['ingredients'].items() if details.type == IngredientType.MAIN)
        base_amount = base_recipe['ingredients'][main_ingredient].amount
        requested_amount = requested_quantities.get(main_ingredient, (base_amount, 'kg'))[0]
        scaling_factor = requested_amount / base_amount

        # Scale ingredients
        for ing_name, ing_details in base_recipe['ingredients'].items():
            # Scaled recipe
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
        """Adjust instructions based on scaling"""
        adjusted = []
        for instruction in instructions:
            if 'cook' in instruction.lower():
                time_adjustment = f" (adjusted for scaled quantities)"
                adjusted.append(instruction + time_adjustment)
            else:
                adjusted.append(instruction)
        return adjusted

    def run_test(self, test_input: str) -> RecipeResult:
        """Run a recipe test based on user input"""
        try:
            # Extract input quantities
            quantities = self.extract_quantities(test_input)

            # Find recipe
            recipe_name = None
            for name in self.recipe_data.keys():
                if name.replace('_', ' ') in test_input.lower():
                    recipe_name = name
                    break

            if not recipe_name:
                raise ValueError("Recipe not found in input")

            # Get base recipe
            base_recipe = self.recipe_data[recipe_name]['base_recipe']

            # Scale recipe
            scaled_recipe, recommended_recipe = self.calculate_recipe_scaling(base_recipe, quantities)

            # Calculate serves
            base_serves = base_recipe['serves']
            main_ingredient = next(ing for ing, details in base_recipe['ingredients'].items() if details.type == IngredientType.MAIN)
            scaling_factor = scaled_recipe['ingredients'][main_ingredient][0] / base_recipe['ingredients'][main_ingredient].amount
            scaled_serves = int(base_serves * scaling_factor)

            # Adjust cooking time
            scaled_time = int(base_recipe['cooking_time'] * scaling_factor)

            # Generate notes
            notes = [base_recipe['notes']]
            if scaling_factor > 2:
                notes.append("Consider cooking in batches for better results")
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
            logger.error(f"Error running test: {str(e)}")
            return RecipeResult(
                recipe_name="unknown",
                input_quantities={},
                scaled_recipe={},
                success=False,
                error_message=str(e)
            )

# Helper function to print results
def print_recipe_result(result: RecipeResult):
    print("\n" + "=" * 50)
    print(f"Recipe: {result.recipe_name}")
    print("=" * 50)
    if result.success:
        print("\nIngredients:")
        for ing, (amount, unit) in result.scaled_recipe['ingredients'].items():
            print(f"- {ing}: {amount:.3f}{unit}")
        print(f"\nCooking Time: {result.cooking_time} minutes")
        print(f"Serves: {result.serves} people")
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")
    else:
        print(f"Test failed: {result.error_message}")

# Main function
if __name__ == "__main__":
    recipe_data = load_recipe_data('data.json')
    tester = ImprovedRecipeTester(recipe_data)

    test_inputs = [
        "I want to make vegetable rice with 10kg rice and 1kg vegetables",
        "Make chicken curry with 2kg chicken",
    ]

    for test_input in test_inputs:
        result = tester.run_test(test_input)
        print_recipe_result(result)
