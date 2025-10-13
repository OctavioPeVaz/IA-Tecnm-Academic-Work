unique = {}
dups = 0
for i, img in enumerate(facesData):
    key = img.tobytes()
    if key in unique:
        dups += 1
    else:
        unique[key] = 1
print("Duplicados exactos detectados:", dups, "Total únicos:", len(unique))