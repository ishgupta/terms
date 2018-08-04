# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:30:58 2018

@author: isgupta

extract important terms from the data

"""

# In[]
import nltk
from nltk.book import FreqDist, bigrams
import string
from itertools import chain
import copy
import os
import docx2txt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from autocorrect import spell

# In[]
    
# the corpus is of documents, and not just the sentences
def extract_phrases(corpus, requested_terms = None, term_freq_threshold = 1, spell_correction = False):
    bigram_docs = extract_bigrams(corpus, spell_correction)
    
    # consolidate bigrams of the coprus, I did not dissolve everything since individual document bigrams are 
    # also needed in further processing
    corpus_bigram = list(chain(*bigram_docs))
    
    # freq of all corpus
    corpus_freq = dict(FreqDist(corpus_bigram).items())
    corpus_bigram = None
    
    # stitch related phrases in the corpus together
    corpus_freq = stitch_related_phrases(bigram_docs, corpus_freq, term_freq_threshold)
    bigram_docs = None
    
    # sort the corpus in descending order of the frequency
    corpus_freq = sorted(corpus_freq.items(), key = lambda x : x[1], reverse = True)
    corpus_freq = dict(corpus_freq)
    
    # initialize stop words
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(" ".join(stop_words).title().split())
    stop_words.extend(string.punctuation)
    stop_words = set(stop_words)
    
    phrases = list(corpus_freq)
    corpus_freq = None
    
    # remove stop words
    phrases = list(map(lambda x : set(x).difference(stop_words), phrases))
    phrases = set(map(lambda x : tuple(x), phrases))
    
    if () in phrases:
        phrases.remove(())
    
    phrases = list(phrases)
    
    if requested_terms is not None:
        phrases = phrases[:requested_terms]
        
    # find nouns
    annotated_data = list(map(lambda x : nltk.pos_tag([*x]) if len(x) > 1 else 'NA', phrases))
    noun_terms = list(filter(lambda x : x is not None and len(x) > 1, map(lambda x : filter_nouns(x), annotated_data)))
    
    #.translate(translator)
    return phrases, noun_terms

# In[]

def extract_important_features_using_tf_idf(data, term_count, consolidate = False):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english')
    
    if consolidate:
        data = [" ".join(data)]
        
    tfidf_matrix =  tf.fit_transform(data)
    feature_names = np.array(tf.get_feature_names())
    #print(feature_names)
    dense = tfidf_matrix.todense()
    phrase_scores = {}
    top_features = {}
    

    # find phrases which have score > 0
    for doc_id in range(0, len(dense)):
        tmp = dense[doc_id].tolist()[0]
        phrase_scores[doc_id] = [pair for pair in zip(range(0, len(tmp)), tmp) if pair[1] > 0]

        # sort the scores in descending order
        sorted_phrase_scores = sorted(phrase_scores[doc_id], key=lambda t: t[1] * -1)
        
        top_features[doc_id] = feature_names[list(map(lambda x : x[0], sorted_phrase_scores[:term_count]))]    
    return top_features


# In[]
    
# corpus is a list of documents
def stitch_related_phrases(bigram_docs, corpus_freq, term_freq_threshold = 1):
    bigrams_to_be_removed = []
    num_of_bigrams_dissolved = 0
    
    for bigram_content in bigram_docs:
        previous_bigram = ()
        for bigram in bigram_content:
            new_bigram = None
            # print(previous_bigram, bigram)
            
            if previous_bigram != () and corpus_freq[bigram] > term_freq_threshold and corpus_freq[bigram] == corpus_freq[previous_bigram]:
                
                #print("{0} #{1} == {2} #{3}".format(previous_bigram, corpus_freq[previous_bigram], bigram, corpus_freq[bigram]))
                new_bigram = (*previous_bigram, bigram[1])
                corpus_freq[new_bigram] = corpus_freq[bigram]
                #print("new bigram: {0} added".format(new_bigram))
                bigrams_to_be_removed.append(bigram)
                bigrams_to_be_removed.append(previous_bigram)
                
                #print("{0} == {1}".format(previous_bigram, bigram))
                num_of_bigrams_dissolved += 2
                # corpus_freq.pop(bigram)
                # corpus_freq.pop(previous_bigram)

            if new_bigram is not None:
                previous_bigram = new_bigram
                # print("previous bigram set to: {0}".format(previous_bigram))
            else:
                previous_bigram = bigram

    # list(filter(lambda x : x[0] in bigrams_to_be_removed, corpus_freq))
    # map(lambda x : corpus_freq.pop(x), bigrams_to_be_removed)
    # [x if x not in bigrams_to_be_removed for x in corpus_freq]
    
    #print("removing bigrams: {0}".format(bigrams_to_be_removed))
    #print("number of bigrams dissolved: {0}".format(num_of_bigrams_dissolved))
    
    for tmp_bigram in bigrams_to_be_removed:
        if tmp_bigram in corpus_freq:
            corpus_freq.pop(tmp_bigram)
    
    
    return corpus_freq


# In[]
    

def filter_nouns(word):
    #print(word)
    nltk_noun_identifiers = [ "NN", "NNS", "NNP", "NNPS" ]
    if word != 'NA':
        #print(word)
        return list(dict(filter(lambda x : x[1] in nltk_noun_identifiers, word)).keys())
    



# In[]
    
def extract_bigrams(list_of_docs, spell_correction = False):
    list_of_bigram_docs = []
    
    for doc in list_of_docs:
        if len(doc) > 0 and doc is not None:
            if not isinstance(doc, list):
                #print("tokenize: {0} for bigrams".format(doc))
                doc = nltk.word_tokenize(doc)
                if spell_correction:
                    doc = list(map(lambda x : spell(x), doc))
                
            list_of_bigram_docs.append(list(bigrams(doc)))
    
    return list_of_bigram_docs

# In[]
# decommissioned
def aggregate_bigram_corpus(list_of_bigram_docs):
    text_bigrams = None
    
    if len(list_of_bigram_docs) > 0:
        text_bigrams = copy.deepcopy(list_of_bigram_docs[0])
        for bigram_doc in list_of_bigram_docs:
            text_bigrams.extend(bigram_doc)
    
    return text_bigrams

# In[]

dir = "data/"
files = os.listdir(dir)
data = []

for file in files:
    print("\n\n\nreading file: {0}".format(file))
    tmp_data = None
    if file.endswith('.doc') or file.endswith('.docx'):
        tmp_data = docx2txt.process(dir + file)
    elif file.endswith('.csv') or  file.endswith('.tsv'):
        tmp_data = pd.read_table(dir + file)
        tmp_data = list(tmp_data['review'])
    if isinstance(tmp_data, list):
        data.extend(tmp_data)
    else:
        data.append(tmp_data)

#data = [list(text1), list(text2)]
important_phrases_from_adjacency_list, noun_terms = extract_phrases(data, requested_terms= 30, term_freq_threshold= 3, spell_correction=True)
print("\n\n\n\nNouns in the filtered data: {0}".format(noun_terms))

print("\n\n\nTerms from TF-IDF", extract_important_features_using_tf_idf(data, 5, True))
print("\n\n\nimportant phrases in data: {0}".format(important_phrases_from_adjacency_list))


