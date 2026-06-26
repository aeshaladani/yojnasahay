girl_keywords = [
    "girl child", "girls only", "women only", "only women", "only girls",
    "female students", "for girls", "for women", "girl student", "women student",
    "kanyashree", "ladli", "beti", "mahila", "stree", "balika",
    "single girl", "daughters", "women empowerment"
]

boy_keywords = [
    "only for boys", "only men", "men only", "male students only", "for boys only"
]

def is_exclusive_for_other_gender(scheme, user_gender):
    if user_gender == "any":
        return False
    text = (
        scheme.get("scheme_name", "") + " " +
        scheme.get("eligibility_raw", "")
    ).lower()
    if user_gender == "male":
        if any(kw in text for kw in girl_keywords):
            return True
    elif user_gender == "female":
        if any(kw in text for kw in boy_keywords):
            return True
    return False

def apply_gender_filter(schemes, user_gender):
    if user_gender == "any":
        return schemes
    before = len(schemes)
    filtered = [s for s in schemes if not is_exclusive_for_other_gender(s, user_gender)]
    after = len(filtered)
    print(f"  Gender filter: {before} → {after} schemes after filtering for {user_gender.upper()}")
    return filtered
