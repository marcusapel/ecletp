# tools/verify_kwargs.py
import json
import inspect

def sig_dict(obj):
    """Return an ordered list of (name, kind, default) for a callable's parameters."""
    sig = inspect.signature(obj)
    out = []
    for name, p in sig.parameters.items():
        # Skip 'self' / 'cls' for methods
        if name in ("self", "cls"):
            continue
        default = None if p.default is inspect._empty else p.default
        out.append({"name": name, "kind": str(p.kind), "default": default})
    return out

def main():
    import resqpy.crs as rqc
    import resqpy.grid as rqq
    import resqpy.property as rqp

    entries = {
        "rqc.Crs.__init__": sig_dict(rqc.Crs.__init__),
        "rqq.RegularGrid.__init__": sig_dict(rqq.RegularGrid.__init__),
        "rqq.Grid.corner_points": sig_dict(rqq.Grid.corner_points),
        "rqp.Property.__init__": sig_dict(rqp.Property.__init__),
        "rqp.Property.from_array": sig_dict(rqp.Property.from_array),
    }

    print("\n--- VERIFIED SIGNATURES ---\n")
    for k, v in entries.items():
        print(k)
        for p in v:
            print(f"  {p['name']}: {p['kind']}, default={p['default']!r}")
        print()

    # Write a machine-readable manifest for CI gating
    with open("verified_kwargs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print("Wrote verified_kwargs_manifest.json")

if __name__ == "__main__":
    main()
