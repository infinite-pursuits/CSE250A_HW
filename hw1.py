import string
import operator

def q1_9a(word_prob_dict):
    sorted_dict_dec = {k:v for k,v in sorted(word_prob_dict.items(), key=lambda item: item[1], reverse=True)[:15]}
    print('TOP 15 WORDS')
    print(list(sorted_dict_dec.keys()))

    sorted_dict_inc = {k: v for k, v in sorted(word_prob_dict.items(), key=lambda item: item[1])[:14]}
    print('\nBOTTOM 14 WORDS')
    print(list(sorted_dict_inc.keys()))

def pick_letter():
    alphabet = list((set(alpha_dict.keys()) - set(incorrect_guesses)-correct_guesses))
    for letter in alphabet:
        alpha_dict[letter] = cal_letter_prob(letter)
    max_pair = max(alpha_dict.items(), key=operator.itemgetter(1))
    return max_pair

def cal_letter_prob(letter):
    empty_ind = find_elem(evidence)
    all_words_prob = 0.0
    total_bayes_denom = 0.0
    bayes_dict = {}
    for word in words:
        p_lw = 0.0
        for v in empty_ind:
            if letter == word[v]:
                p_lw = 1.0
                break

        bayes_num, p_ew = bayes(word)
        total_bayes_denom += bayes_num
        if p_ew>0.0 and p_lw>0.0:
            bayes_dict[word] = bayes_num

    for word, num in bayes_dict.items():
        bayes_term = num/total_bayes_denom
        all_words_prob += bayes_term
    return all_words_prob

def bayes(word):
    p_w = word_prob_dict[word]
    p_ew = evidence_routine(word)
    num = p_ew * p_w
    return num, p_ew

def evidence_routine(word):
    p_ew = 1.0
    for i in range(5):
        if ((not evidence[i]) and (word[i] not in incorrect_guesses) and (word[i] not in correct_guesses))\
                or (evidence[i] and (evidence[i]==word[i])):
            continue
        p_ew = 0.0
        break
    return p_ew


def find_elem(evidenceorword, dict_flag =1,string = ''):
    ind_list = []
    if dict_flag:
        for k,v in evidenceorword.items():
            if v==string:
                ind_list.append(k)
    else:
        ind_list = [i for i,letter in enumerate(evidenceorword) if letter==string]
    return ind_list

fn = '/Users/chhaviyadav/Desktop/HW_2020_Fall/250A/hw1/hw1_word_counts_05.txt'
word_count_dict = {}
word_prob_dict = {}
total = 0.0
sum = 0.0

with open(fn) as f:
    lines = f.readlines()
    for line in lines:
        val = line[:5]
        num = int(line[6:-1])
        total += num
        word_count_dict[val] = num

for k,v in word_count_dict.items():
    prob = float(v) / float(total)
    word_prob_dict[k] = prob
    sum += prob

print(sum)
q1_9a(word_prob_dict)

alpha_dict = dict.fromkeys(list(string.ascii_uppercase), 0.0)
evidence = {0:'',1:'',2:'',3:'',4:''} #edit this for testing
incorrect_guesses = [] #edit this for testing
correct_guesses = set(evidence.values())
words = list(word_prob_dict.keys())

print('\nBEST PICK:')
print(pick_letter())