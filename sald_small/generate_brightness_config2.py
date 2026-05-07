import sys
import json
import copy


input_file = sys.argv[1]

with open(input_file, "r", encoding="utf-8") as f:
    base_config = json.load(f)

for r in [1,5,10,15,20]:
    new_config = copy.deepcopy(base_config)
    new_config["r"] = r    

    filename = f"config_brightness_r{r}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)

        

                
