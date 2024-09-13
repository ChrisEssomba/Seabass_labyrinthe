#Allow to adjust the format of the bounding position for older versions of the code
import re

# Paths
old_format_path = r"D:/Chris/2023_PFG_Daniel_Areñas_Mayoral/boxes_positions.txt"
new_format_path = r"D:/Chris/2023_PFG_Daniel_Areñas_Mayoral/boxes_positions_new.txt"

# Function to convert old format to new format
def convert_format(old_file_path, new_file_path):
    with open(old_file_path, 'r') as old_file, open(new_file_path, 'w') as new_file:
        lines = old_file.readlines()
        
        current_frame = None
        for line in lines:
            line = line.strip()
            
            # Identify frame line
            if line.startswith("Frame"):
                if current_frame is not None:
                    new_file.write("\n")
                current_frame = line
                new_file.write(current_frame + "\n")
                
            # Extract and reformat bounding boxes
            elif line and not line.startswith("Frame"):
                # Find BBox coordinates using regex
                bbox_match = re.search(r'BBox: \(([^)]+)\)', line)
                if bbox_match:
                    bbox_str = bbox_match.group(1)
                    x_min, y_min, x_max, y_max = map(float, bbox_str.split(', '))
                    # Write in new format
                    new_file.write(f"x_min={x_min},y_min={y_min},x_max={x_max},y_max={y_max}\n")

# Convert the file
convert_format(old_format_path, new_format_path)
