import json, glob
RAW = r"D:\Bits_course\Dis\Dataset\Fusion360"
seen = 0
for p in glob.glob(RAW + r"\**\assembly.json", recursive=True):
    joints = (json.load(open(p)).get("joints") or {})
    if not joints: continue
    for jid, j in joints.items():
        g1, g2 = j.get("geometry_or_origin_one"), j.get("geometry_or_origin_two")
        def descr(g):
            if not isinstance(g, dict): return "MISSING"
            e = g.get("entity_one")
            has_axis = "primary_axis_vector" in g
            if isinstance(e, dict):
                return f"entity={e.get('type')} idx={e.get('index')} poe={'point_on_entity' in e} axis={has_axis}"
            return f"NO-ENTITY axis={has_axis} keys={list(g.keys())}"
        print(j.get("joint_motion",{}).get("joint_type"), "| one:", descr(g1), "| two:", descr(g2))
    seen += 1
    if seen >= 8: break