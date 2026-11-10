# install pyyaml
# pip install pyyaml

from pprint import pprint

import yaml

# consume YAML file
with open("data.yaml", "r") as file:
    data = yaml.safe_load(file)

# pprint(data, indent=2,width=50)
# print("app name:",data["app_name"])
# print("version:",data["version"])
# print("score:",data["score"])
# print("is production:",data["is_production"])
# print("connection string:",data["connection_string"])
# print("supported languages:",data["supported_languages"])

# db_info = data.get("database", {})
# print("database info:", db_info)
# print("database host:", db_info.get("host"))
# print("database port:", db_info.get("port"))
# print("database username:", db_info.get("username"))
# print("database password:", db_info.get("password"))

# missingness_rules = data.get("missingness_rules", {})
# print("missingness rules:", missingness_rules)
# knn_imputer = missingness_rules.get("knn_imputer", {})
# print("knn imputer n_neighbors:", knn_imputer.get("n_neighbors","<not set>"))
# print("knn imputer weights:", knn_imputer.get("weights","<not set>"))

# print("indicator flagging cols:", missingness_rules.get("indicator_flagging_cols","<not set>"))
# print("mnar sentinel value:", missingness_rules.get("mnar_sentinel_value","<not set>"))

# info = data.get("nirmal", {})
# pprint(info)
info_komal = data.get("komal", {})
pprint(info_komal)