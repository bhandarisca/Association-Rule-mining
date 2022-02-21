# %% import dataframe from pickle file
import pandas as pd

df = pd.read_pickle("Uk.pkl")

df.head()


# %% convert dataframe to invoice-based transactional format
dataset = []
for inv, gdf in df.groupby("InvoiceNo"):
    items = gdf["Description"].tolist()
    dataset.append(items)



# %% apply apriori algorithm to find frequent items and association rules


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

transactions = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(
    transactions, min_support=0.02, use_colnames=True
)
rules = association_rules(frequent_itemsets, min_threshold=0.5)



# %% count of frequent itemsets that have more then 1/2/3 items,
# and the frequent itemsets that has the most items
length = frequent_itemsets["itemsets"].apply(len)

frequent_itemsets["length"] = length

frequent_itemsets
#%%
print((frequent_itemsets["length"] > 1).sum())
print((frequent_itemsets["length"] > 2).sum())
print((frequent_itemsets["length"] > 3).sum())

#%%
most_items = frequent_itemsets["length"] == frequent_itemsets["length"].max()
frequent_itemsets[most_items]

# %% top 10 lift association rules

rules.sort_values("lift").head(10)

# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot support vs lift
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("Support vs lift")

# %%
