import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge')
os.makedirs(base_dir, exist_ok=True)

crops = {
    'coffee': "Coffee requires a well-drained, deeply cultivated soil and moderate warmth. Pruning is essential to maximize crop yields. Shade management during extreme dry seasons is highly recommended.",
    'maize': "Maize demands frequent fertilization, especially rich nitrogen. Sow seeds deeply when soil temperatures are warm. Fall armyworm attacks are common; use preventive organic sprays.",
    'mango': "Mango trees thrive in tropical climates with a marked dry season for flowering. Avoid excessive watering during the dormant phase. Fungal rot is a key hazard during wet seasons.",
    'cotton': "Cotton is deeply reliant on heavy soils and long sunny periods. High phosphorus early in the cycle encourages deep root development. Watch eagerly for bollworm infestations."
}

for crop, info in crops.items():
    with open(os.path.join(base_dir, f"{crop}.txt"), 'w', encoding='utf-8') as f:
        f.write(info)

print("Knowledge base created.")
