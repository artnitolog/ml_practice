from collections import Counter

# Counter was allowed in telegram chat


def check_first_sentence_is_second(s1, s2):
    count = Counter(s1.split())
    count.subtract(s2.split())
    return not -count
