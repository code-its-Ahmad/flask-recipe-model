from flask import Flask, request, jsonify
from main import ImprovedRecipeTester

app = Flask(__name__)

# Initialize the recipe tester
tester = ImprovedRecipeTester()

@app.route('/')
def home():
    return "Welcome to the Recipe API!"

@app.route('/scale_recipe', methods=['POST'])
def scale_recipe():
    try:
        data = request.json
        test_input = data.get('test_input')

        if not test_input:
            return jsonify({"error": "No input provided"}), 400

        result = tester.run_test(test_input)
        result_dict = {
            "recipe_name": result.recipe_name,
            "input_quantities": result.input_quantities,
            "scaled_recipe": result.scaled_recipe,
            "recommended_recipe": result.recommended_recipe,
            "success": result.success,
            "error_message": result.error_message,
            "cooking_time": result.cooking_time,
            "serves": result.serves,
            "notes": result.notes,
        }
        return jsonify(result_dict), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
