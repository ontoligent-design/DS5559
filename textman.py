#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

textman.py
Created on Wed Dec 12 22:05:54 2018
@author: rca2t

* Converts work done in First Foray notebooks into functions. Next
step is to create an object model.
* Containers above sentence and token should be user defined, 
but sentences and tokens should not be, 
* POS tagging applies to DOCTERM (i.e. tokens), not to TERM (vocab)
* We create DOCTERM from the source data and derive the TERM and DOC tables
from it. We use the derived tables to park informaton where it goes. We 
use the model to understand what we are doing!

"""

import pandas as pd
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np


def import_source(src_file, start_line, end_line, col_name='line', id_name='line_id', strip=True):
    df = pd.DataFrame({col_name:open(src_file,'r').readlines()})
    df = df.loc[start_line:end_line]
    df.index.name = id_name
    if strip:
        df[col_name] = df[col_name].str.strip()
    return df

def group_by_milestone(df, div_name, div_pat, src_idx, src_col, tmp_col='div_idx', id_suffix='_id'):
    df[div_name] = df[src_col].str.match(div_pat)
    df2 = df.loc[df[div_name], src_col].copy().reset_index(drop=True).to_frame()\
        .rename(columns={src_col:div_name})
    df2.index.name = div_name + id_suffix
    df[tmp_col] = None
    df[tmp_col] = df[df[div_name]].apply(lambda x: x.index)
    df[tmp_col] = df[tmp_col].ffill()
    df[tmp_col] = df[tmp_col].astype('int')
    df2[div_name] = df.groupby(tmp_col)[src_col].apply(lambda x: '\n'.join(x[:]))\
        .to_frame().reset_index(drop=True)
    df2.index.name = div_name + id_suffix
    return df2

def split_by_delimitter(df, div_name=None, div_pat=None, src_col=None, join_pat='\n', id_suffix='_id'):
    df2 = df[src_col].str.split(div_pat, expand=True).stack().to_frame()\
        .rename(columns={0:div_name}).copy()
    df2.index.names = df.index.names + [div_name + id_suffix]
    df2[div_name] = df2[div_name].str.replace(join_pat, ' ')
    df2 = df2[~df2[div_name].str.match(r'^\s*$')]
    return df2

# Save this for object
# Include POS recognization here since it is more efficient
def tokenize(df, with_sent=True):
    pass

def normalize_tokens(df, token_col='token'):
    df['token_norm'] = df[token_col].str.lower().str.strip()
    return df

# DO THIS BEFORE STOPWORD REMOVAL
# Maybe do when tokenizing by sentence'
# However, POS sucks -- check out 'ahab'
def add_pos_to_tokens(tokens, idx=['chap_id','para_id','sent_id'], token_col='token'):
    df = tokens.groupby(idx).token.apply(lambda x: nltk.pos_tag(x.tolist()))\
        .apply(pd.Series).stack()\
        .to_frame().reset_index()\
        .rename(columns={'level_{}'.format(len(idx)):'token_id', 0:'pos'})\
        .set_index(idx + ['token_id'])
    tokens['pos'] = df.pos.apply(lambda x: x[1])
    return tokens

def create_vocab(df, col='token_norm'):
    terms = df[col].value_counts()\
        .to_frame().reset_index()\
        .rename(columns={'index':'term',col:'n'})\
        .sort_values('term').reset_index(drop=True)
    terms.index.name = 'term_id'
    terms['f'] = terms.n.div(terms.n.sum())
    return terms

def add_stems_to_vocab(df):
    ps = PorterStemmer()
    vocab['stem'] = vocab['term'].apply(lambda x: ps.stem(x))
    return vocab

def link_tokens_to_vocab(tokens, vocab, drop=False):
    tokens['term_id'] = tokens['token_norm'].map(vocab.reset_index().set_index('term').term_id)
    if drop:
        del(tokens['token_norm'])
    return tokens

def identify_stopwords(vocab):
    sw = set(stopwords.words('english'))
    vocab['sw'] = vocab.term.apply(lambda x: 
        x in sw or len(x) < 2 or not x.isalpha())
    return vocab

def remove_stopwords(df, vocab, term_id_col='term_id'):
    df = df[df[term_id_col].isin(vocab[~vocab.sw].index.values)]
    return df

def create_doc_table(tokens, index=['chap_id', 'para_id']):
    doc = tokens.groupby(index).term_id.count()\
        .to_frame().rename(columns={'term_id':'n'})
    return doc

def create_bow(tokens, idx, index_name='doc_id'):
    col = idx[-1]
    bow = tokens.groupby(idx)[col].count()\
        .to_frame().rename(columns={col:'n'})
    if index_name:
        bow.index.name = index_name
    return bow

def create_dtm(bow, fill_val=0):
    dtm = bow.unstack().fillna(fill_val)
    dtm.columns = dtm.columns.droplevel(0)
    return dtm

def compute_term_freq(dtm, vocab):
    dtm_tf = dtm.apply(lambda x: x / x.sum(), 1)
    vocab['tf_sum'] = dtm_tf.sum()
    return dtm, vocab

def compute_inv_doc_freq(dtm, vocab):
    N = len(dtm.index)
    dtm_idf = dtm.apply(lambda x: N / x[x > 0].count())
    vocab['idf'] = dtm_idf
    return dtm_idf, vocab

def compute_tfidf(dtm, vocab, doc, bow, sw=False):
    N = len(dtm.index)
    dtm_tfidf = dtm.apply(lambda row: row / row.sum(), 1)\
        .apply(lambda col: col * np.log(N/col[col > 0].count()))
    vocab['tfidf_sum'] = dtm_tfidf.sum()
    doc['tfidf_sum'] = dtm_tfidf.sum(1)
    bow['tfidf'] = dtm_tfidf.stack().to_frame().rename(columns={0:'tfidf'})
    return dtm_tfidf, vocab, doc, bow

def compute_tfidh():
    pass

#def compute_tfidf2(dtmtf, vocab, bow, sw=False):
#    if sw:
#        V = vocab
#    else:
#        V = vocab[~vocab.sw]
#    dtm_tfidf = dtmtf.apply(lambda x: x * np.log(V.idf), 1)
#    #dtm_tfidf = dtmtf.multiply(np.log(V.idf), 1)
#    vocab['tfidf_sum1'] = dtm_tfidf.sum()
#    bow['tfidf1'] = dtm_tfidf.stack().to_frame().rename(columns={0:'tfidf'})
#    return dtm_tfidf, vocab, bow

def get_term_id(vocab, term):
    term_id = vocab[vocab.term==term].index[0] 
    return term_id

def get_term(vocab, term_id):
    term = vocab.loc[term_id].term
    return term

# Put these in another class (use SA?)

def put_to_db(db, df, table_name, index=True, if_exists='replace'):
    r = df.to_sql(table_name, db, index=index, if_exists=if_exists)
    return r
    
def get_from_db(db, table_name):
    df = pd.read_sql("SELECT * FROM {}".format(table_name), db)
    return df


if __name__ == '__main__':
    
#    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    
    
    pwd = '/Users/rca2t/Dropbox/Courses/DSI/DS5559/course-repo-od'
    db = sqlite3.connect(pwd + '/moby.db')
    
    config = dict(
        clips = dict(
            src_file = pwd + '/moby.txt',
            start_line = 318,
            end_line = 23238           
        ),
        chap = dict(
            div_name = 'chap',
            div_pat = r'^(?:ETYMOLOGY|CHAPTER \d+|Epilog)',
            src_idx = 'line_id',
            src_col = 'line'
        ),
        para = dict(
            div_name = 'para',
            div_pat = r'\n\n+',
            src_col = 'chap'
        ),
                
        sent = dict(
            div_name = 'sent',
            div_pat = r'(?:[":;.?!\(\)]|--)',
            src_col = 'para',
            join_pat = ' '
       ),
       token = dict(
           div_name = 'token',
           div_pat = r'\W+',
           src_col = 'sent',
           join_pat = ' '
       )
    )

    print("SRC")
    src = import_source(**config['clips'])
    
    print("CHAP")
    chaps = group_by_milestone(src, **config['chap'])
    
    print("PARA")
    paras = split_by_delimitter(chaps, **config['para'])
    
    print("SENT")
    sents = split_by_delimitter(paras, **config['sent'])
    
    print("TOKENS")
    tokens = split_by_delimitter(sents, **config['token'])
    
    print("VOCAB")
    tokens = normalize_tokens(tokens)
    tokens = add_pos_to_tokens(tokens)
    vocab = create_vocab(tokens)
    vocab = add_stems_to_vocab(vocab)
    vocab = identify_stopwords(vocab)
    tokens = link_tokens_to_vocab(tokens, vocab, drop=True)
#    tokens = remove_stopwords(tokens, vocab)
    
    print("DOC")
    doc = create_doc_table(tokens, ['chap_id', 'para_id'])
    
    print("BOW")
    bow = create_bow(tokens, ['chap_id', 'para_id', 'term_id'])

    print("DTM")
    dtm = create_dtm(bow)


    print("TFIDF")
    tfidf, vocab, doc, bow = compute_tfidf(dtm, vocab, doc, bow)
    
#    print("TFIDF2")
#    dtmtf, vocab = compute_term_freq(dtm, vocab)
#    dtmidf, vocab = compute_inv_doc_freq(dtm, vocab) # DEPENDENCY ON DOC UNIT!
#    tfidf2, vocab, bow = compute_tfidf2(dtmidf, vocab, bow)
    
    print("PLOTS")
    tfidf.sum(1).plot(figsize=(10,2))    
    tfidf[[get_term_id(vocab, 'ahab'),get_term_id(vocab, 'whale')]].plot(figsize=(10,2))

#    put_to_db(db, f0, 'f0')
#    put_to_db(db, f1, 'f1_chap')
#    put_to_db(db, f2, 'f2_para')
#    put_to_db(db, f3, 'f3_sent')
#    put_to_db(db, tokens, 'docterm')
    
