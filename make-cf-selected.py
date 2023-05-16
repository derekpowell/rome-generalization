import pandas as pd
import json
import random

df = pd.read_csv("counterfact/counterfact-selected-relations.csv")

with open('counterfact/counterfact.json') as json_file:
   json_data = json.load(json_file)

qual_items = pd.read_csv("counterfact/selected-items.csv")


## prepare new counterfact

rel_ids = df.relation_id.to_list()

json_out = []

for j in json_data:
    if j["requested_rewrite"]["relation_id"] in rel_ids: json_out.append(j)

print(len(json_out), "items in selected relations")


random.seed(2346)

json_sampled = random.sample([j for j in json_out if j["case_id"] not in qual_items.case_id], k = 750)
print("500 sampled validation items")
print("250 sampled experimentation items")


json_qual = []
for j in json_data:
    if j["case_id"] in qual_items.case_id: json_qual.append(j)
    
print(len(json_qual), "qualitative test set items")

with open("data/counterfact-selected.json", "w") as file:
    json.dump(json_out, file, indent=4)

with open("data/counterfact-selected-train250.json", "w") as file:
    json.dump(json_sampled[500:], file, indent=4)
    
with open("data/counterfact-selected-valid500.json", "w") as file:
    json.dump(json_sampled[:500], file, indent=4)
    
with open("data/counterfact-selected-qual.json", "w") as file:
    json.dump(json_qual, file, indent=4)