import re
import json
import argparse

def extract_json_from_file(input_path, output_path):
    """
    Extract JSON blocks from a given text file and save them to an output file.

    :param input_path: Path to the input text file.
    :param output_path: Path to the output text file.
    """
    json_objects = []

    # Regular expression to match JSON-like structures
    json_pattern = re.compile(r'\{(?:[^{}]*|(?:\{[^{}]*\}))*\}', re.DOTALL)

    try:
        with open(input_path, 'r') as file:
            content = file.read()

            res = extract_json_from_text(content)
            print(res)

            # Find all JSON matches in the content
            matches = json_pattern.findall(content)

            for match in matches:
                try:
                    # Parse the JSON string to ensure it is valid
                    json_obj = json.loads(match)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Invalid JSON detected and skipped: {match[:30]}...")

        # Save the extracted JSON objects as strings without quotes
        with open(output_path, 'w') as output_file:
            for json_obj in json_objects:
                json_str = json.dumps(json_obj)
                json_str = json_str.replace('"', '\\"')
                output_file.write(json_str + '\n')  # Keeping the string valid JSON

        print(f"Extracted JSON objects have been saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_json_from_text(content):

    json_objects = []

    # Regular expression to match JSON-like structures
    json_pattern = re.compile(r'\{(?:[^{}]*|(?:\{[^{}]*\}))*\}', re.DOTALL)

    # Find all JSON matches in the content
    matches = json_pattern.findall(content)

    for match in matches:
        try:
            # Parse the JSON string to ensure it is valid
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            print(f"Invalid JSON detected and skipped: {match[:30]}...")

    return json_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract JSON blocks from a text file.")
    parser.add_argument("input_path", type=str, help="Path to the input text file.")
    parser.add_argument("output_path", type=str, help="Path to the output text file.")

    args = parser.parse_args()

    extract_json_from_file(args.input_path, args.output_path)
