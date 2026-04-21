import sys
import json
import copy


input_file = sys.argv[1]

with open(input_file, "r", encoding="utf-8") as f:
    base_config = json.load(f)

r = 1.0
for i in range(-4, 5):
    b = i / 10.0

    new_config = copy.deepcopy(base_config)
    new_config["r"] = r    
    new_config["target_mean"] = b

    filename = f"config_r{r}_mean{b:.1f}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(new_config, f, ensure_ascii=False, indent=4)

        

                
