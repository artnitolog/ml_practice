from collections import defaultdict


def get_new_dictionary(input_dict_name, output_dict_name):
    with open(input_dict_name, 'r') as fin, \
         open(output_dict_name, 'w') as fout:
        dragonic_human = defaultdict(list)
        fin.readline()
        for line in fin:
            hum_word, dr_words = line.split(' - ')
            for dr_word in dr_words.strip().split(', '):
                dragonic_human[dr_word] += [hum_word]
        fout.write(str(len(dragonic_human)) + '\n')
        for dr_word, hum_words in sorted(dragonic_human.items()):
            fout.write(dr_word + ' - ' + ', '.join(sorted(hum_words)) + '\n')
