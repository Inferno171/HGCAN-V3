import json, glob, numpy as np
RAW = r"D:\Bits_course\Dis\Dataset\Fusion360"
for p in glob.glob(RAW + r"\**\assembly.json", recursive=True):
    doc = json.load(open(p)); J = doc.get("joints") or {}
    if not J: continue
    j = next(iter(J.values()))
    for side in ("geometry_or_origin_one", "geometry_or_origin_two"):
        e = (j.get(side) or {}).get("entity_one") or {}
        poe = e.get("point_on_entity"); bb = e.get("bounding_box")
        print(side, "surface:", e.get("surface_type"))
        print("  point_on_entity:", None if not poe else [round(poe[k],3) for k in "xyz"])
        if bb: print("  entity bbox max:", [round(bb["max_point"][k],3) for k in "xyz"])
    break