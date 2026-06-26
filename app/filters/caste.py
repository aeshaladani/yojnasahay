caste_keywords = {
    "sc": ["scheduled caste", "sc students", "sc category", "sc/st"],
    "st": ["scheduled tribe", "st students", "st category"],
    "obc": ["other backward", "obc students", "obc category"],
    "general": ["general category", "general merit", "non-reserved"],
}

def is_exclusive_for_other_caste(scheme, user_caste):
    if user_caste == "any":
        return False
    text = (
        scheme.get("scheme_name", "") + " " +
        scheme.get("eligibility_raw", "")
    ).lower()
    for caste, keywords in caste_keywords.items():
        if caste == user_caste:
            continue
        if any(kw in text for kw in keywords):
            if user_caste == "general" and caste in ["sc", "st", "obc"]:
                return True
            if user_caste == "sc" and caste in ["st", "obc"]:
                return True
            if user_caste == "st" and caste in ["sc", "obc"]:
                return True
            if user_caste == "obc" and caste in ["sc", "st"]:
                return True
    return False

def apply_caste_filter(schemes, user_caste):
    if user_caste == "any":
        return schemes
    before = len(schemes)
    filtered = [s for s in schemes if not is_exclusive_for_other_caste(s, user_caste)]
    after = len(filtered)
    print(f"  Caste filter: {before} → {after} schemes after filtering for {user_caste.upper()}")
    return filtered
