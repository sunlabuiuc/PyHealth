def normalize_icd9cm(code: str):
    """Normalize ICD9CM code."""
    if code.startswith("E"):
        assert len(code) >= 4
        if len(code) == 4:
            return code
        return code[:4] + "." + code[4:]
    else:
        assert len(code) >= 3
        if len(code) == 3:
            return code
        return code[:3] + "." + code[3:]


def normalize_icd9proc(code: str):
    """Normalize ICD9PROC code."""
    assert len(code) >= 2
    if len(code) == 2:
        return code
    return code[:2] + "." + code[2:]


def normalize_icd10cm(code: str):
    """Normalize ICD10CM code."""
    assert len(code) >= 3
    if len(code) == 3:
        return code
    return code[:3] + "." + code[3:]
