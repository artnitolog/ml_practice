def prefix(s):
    res = [0]
    for cur in s[1:]:
        k = res[-1]
        while k > 0 and cur != s[k]:
            k = res[k - 1]
        res += [(cur == s[k]) + k]
    return res


def find_max_substring_occurrence(input_string):
    period = len(input_string) - prefix(input_string)[-1]
    if len(input_string) % period == 0:
        return len(input_string) // period
    else:
        return 1
