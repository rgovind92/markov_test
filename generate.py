from collections import defaultdict, Counter
import random
import sys
import nltk
from pdb import set_trace

moby = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
moby = ' '.join(moby)


# This is the length of the "state" the current character is predicted from.
# For Markov chains with memory, this is the "order" of the chain. For n-grams,
# n is STATE_LEN+1 since it includes the predicted character as well.
STATE_LEN = 2

data = moby
model = defaultdict(Counter)

set_trace()

print('Learning model...')
for i in range(len(data) - STATE_LEN):
    state = data[i:i + STATE_LEN]
    next = data[i + STATE_LEN]
    model[state][next] += 1

set_trace()

print('Sampling...')
state = random.choice(list(model))
out = list(state)
for i in range(400):
    out.extend(random.choices(list(model[state]), model[state].values()))
    state = state[1:] + out[-1]

set_trace()
print(''.join(out))
