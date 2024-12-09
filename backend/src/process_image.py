import sys
import os

def process_image(file_path):
    if not os.path.exists(file_path):
        return "Error: File not found."

    # For demonstration, we'll just return the file size and name
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    
    return f"Image '{file_name}' with size {file_size} bytes has been uploaded. Click next button to process."

if __name__ == "__main__":
    # The first argument is the script name, the second is the file path
    if len(sys.argv) < 2:
        print("Error: No img path provided.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = process_image(file_path)
    print("result: ", result)  # Output the result
