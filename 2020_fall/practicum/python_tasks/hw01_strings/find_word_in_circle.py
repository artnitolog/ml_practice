def cut(s, n):
    if n == 0:
        return ''
    chunks = {s[i:i+n] for i in range(0, len(s), n)}
    if len(chunks) > 2:
        return ''
    elif len(chunks) == 2:
        end, beg = sorted(chunks, key=len)
        if beg.find(end) != 0:
            return ''
        return end
    else:
        return chunks.pop()


def find_word_in_circle(circle, word):
    word = cut(word, len(circle))
    if word == '':
        return -1
    direct_idx = (circle * 2).find(word)
    if direct_idx != -1:
        return direct_idx, 1
    inverse_idx = (circle[::-1] * 2).find(word)
    if inverse_idx != -1:
        return len(circle) - inverse_idx - 1, -1
    return -1
