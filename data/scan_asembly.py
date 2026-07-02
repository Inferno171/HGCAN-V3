import json, glob
RAW = r"D:\Bits_course\Dis\Dataset\Fusion360"
flat = pc_edges = no_contacts = total = 0
for p in glob.glob(RAW + r"\**\assembly.json", recursive=True):
    d = json.load(open(p))
    if not d.get("joints"): continue
    total += 1
    occs = d["occurrences"]; wb = {u for u,o in occs.items() if (o or {}).get("bodies")}
    par = {}
    def w(n,pa):
        for u,ch in (n or {}).items(): par[u]=pa; w(ch,u)
    w((d.get("tree") or {}).get("root",{}), None)
    pc = sum(1 for u in wb if par.get(u) in wb)
    pc_edges += (pc > 0)
    if pc == 0: flat += 1
    if not d.get("contacts"): no_contacts += 1
print(f"jointed assemblies: {total}")
print(f"  with >=1 body-to-body parent-child tree edge: {pc_edges}")
print(f"  completely flat (no such edge): {flat}")
print(f"  with null/empty contacts: {no_contacts}")