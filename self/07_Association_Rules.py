# ---------- Q7: Apriori association rules on SYNTHETIC basket data ----------
# NOTE: these baskets are simulated for practice, not real shopping data.
import random
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Build a reproducible synthetic transaction list
random.seed(42)
items = ["Bread", "Milk", "Butter", "Eggs", "Jam", "Coffee"]

transactions = []
for _ in range(40):
    basket = set()
    if random.random() < 0.70: basket.add("Bread")
    if random.random() < 0.55: basket.add("Milk")
    if "Bread" in basket and random.random() < 0.80: basket.add("Butter")  # pattern
    if random.random() < 0.40: basket.add("Eggs")
    if "Bread" in basket and random.random() < 0.60: basket.add("Jam")     # pattern
    if random.random() < 0.35: basket.add("Coffee")
    transactions.append(sorted(basket))

print("Sample transactions:", transactions[:5])

# 2. One-hot encode the transactions
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(transactions).transform(transactions),
                      columns=te.columns_)
print("\nEncoded preview:\n", onehot.head())

# 3-4. Run Apriori at two support thresholds and generate rules
for min_sup in [0.20, 0.35]:
    freq = apriori(onehot, min_support=min_sup, use_colnames=True)
    freq["length"] = freq["itemsets"].apply(len)

    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    rules = rules[rules["lift"] > 1]   # keep only positive associations

    print(f"\n===== min_support = {min_sup} =====")
    print("Frequent itemsets:", len(freq), "| Rules kept:", len(rules))
    if len(rules) > 0:
        print(rules.sort_values("lift", ascending=False)
                   .head(5)[["antecedents", "consequents",
                             "support", "confidence", "lift"]].round(3))
