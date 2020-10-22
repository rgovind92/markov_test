#!/usr/bin/python

import math
import pickle
import pdb
import nltk
import random
from collections import defaultdict, Counter

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

book = 'shakespeare-macbeth.txt'

data = ' '.join(nltk.corpus.gutenberg.words(book))

#book = 'rubbish'

#data = ''.join([accepted_chars[random.randint(0, 26)] for _ in range(99999)])

#print(data)

def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation, 
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def train():
    """ Write a simple model as a pickle file """
    k = len(accepted_chars)

    #pdb.set_trace()
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    
    #counts = [[10 for i in range(k)] for i in range(k)]

    # Count transitions from big text file, taken 
    # from http://norvig.com/spell-correct.html
    
    counts = defaultdict(Counter)

    '''for a, b in ngram(2, data):
        counts[pos[a]][pos[b]] += 1
    '''

    for a in ngram(3, data):
        counts[a[0:2]][a[2]] += 1
    
    #pdb.set_trace()

    # Normalize the counts so that they become log probabilities.  
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    '''for k in counts:
        s = float(sum(counts[k]))
        for j in range(len(counts[k])):
            counts[k][j] = math.log(counts[k][j] / s)
    '''
    
    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    '''good_probs = [avg_transition_prob(l, counts) for l in open('good.txt')]
    bad_probs = [avg_transition_prob(l, counts) for l in open('bad.txt')]

    pdb.set_trace()

    # Assert that we actually are capable of detecting the junk.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    '''
    #pdb.set_trace()
    pickle.dump({'mat': dict(counts) }, open('gib_model_' + book + '.pki', 'wb'))

def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    #pdb.set_trace()
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

import numpy

def train(data):
    #counts = defaultdict(lambda: [10 for _ in range(27)])
    model = defaultdict(Counter)

    counts = [[10 for i in range(27)] for i in range(27)]

    pdb.set_trace()

    for a in ngram(3, data):
        #counts[a[0:2]][pos[a[2]]] += 1
        model[a[0:2]][a[2]] += 1    

    for a, b in ngram(2, data):
        counts[pos[a]][pos[b]] += 1

    '''for k in counts:
        s = float(sum(counts[k]))
        for j in range(len(counts[k])):
            counts[k][j] = math.log(counts[k][j] / s)
    '''
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in range(len(row)):
            row[j] = math.log(row[j] / s)
 
    pdb.set_trace()

    return (counts, model)

def generate(model):
    out = numpy.random.choice(list(model.keys()))
    current = out

    for i in range(400):
      next = random.choices(list(model[current]), model[current].values())[0]
      #out += accepted_chars[numpy.random.choice(range(27), p=[math.exp(_) for _ in model['mat'][pos[out[-1]]]])]
      out += next
      current = out[i+1:]

    return out
    '''for a, b in ngram(2, data):
        counts[pos[a]][pos[b]] += 1
    '''

    '''for a in ngram(3, data):
        model[a[0:2]][a[2]] += 1
    '''
    #pdb.set_trace()

    #model = pickle.load(open('gib_model_' + book + '.pki', 'rb'))['mat']

    '''out = numpy.random.choice(list(model.keys()))
    current = out

    for i in range(400):
      next = random.choices(list(model[current]), model[current].values())[0]
      #out += accepted_chars[numpy.random.choice(range(27), p=[math.exp(_) for _ in model['mat'][pos[out[-1]]]])]
      out += next
      current = out[i+1:]

    print(out)
    '''

if __name__ == '__main__':
    #train()
    books = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

    data_eng = ' '.join(nltk.corpus.gutenberg.words(books[0]))
    data_pol = open('zoltan', 'r').read()

    data_eng = open('passwords/Leaked-Databases/muslimMatch.txt').read().replace('\n', ' ')
    data_pol = open('passwords/Leaked-Databases/myspace.txt').read().replace('\n', ' ')
    
    counts_eng, model_eng = train(data_eng)
    gen_eng = generate(model_eng)

    counts_pol, model_pol = train(data_pol)
    gen_pol = generate(model_pol)

    #print(d[0] + '\n')
    #print(gen)
    tot = 0
    i = 0
    for w in gen_eng.split():
        if len(w) > 2:
            i += 1
            score_eng = avg_transition_prob(w, counts_eng)
            print(w + '    ' + str(score_eng))
            tot += score_eng
    print('Sum: ' + str(tot))   

    print('\n\n\n\n\n')

    tot = 0
    i = 0
    for w in gen_eng.split():
        if len(w) > 2:
            i += 1
            score_pol = avg_transition_prob(w, counts_pol)
            tot += score_pol
            print(w + '    ' + str(score_pol))
    print('Sum: ' + str(tot))


    
    
