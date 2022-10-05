def normalize_icd9(code):
    """Normalize ICD9 code"""
    if code.startswith('E'):
        assert len(code) >= 4
        if len(code) == 4:
            return code
        return code[:4] + '.' + code[4:]
    else:
        assert len(code) >= 3
        if len(code) == 3:
            return code
        return code[:3] + '.' + code[3:]
