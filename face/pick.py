import os
import joblib
with open(str(os.getcwd()+"/model_train/embeddings/embeddings.pkl"), "rb") as f:
    (saved_embeds, names) = joblib.load(f)

for n,s in zip(names,saved_embeds):
	print(n,s)

print(names)
print(set(names))