from pydantic import BaseModel
import pydantic as pydantic_v1


def deep_union_pydantic_dicts(dict1, dict2):
    # Example simple deep merge
    if not dict1:
        return dict2
    if not dict2:
        return dict1
    result = dict1.copy()
    for key, val in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_union_pydantic_dicts(result[key], val)
        else:
            result[key] = val
    return result
