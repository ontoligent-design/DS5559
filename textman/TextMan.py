#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:34:50 2018

@author: rca2t
"""

import pandas as pd
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np


class Corpus(object):
    
    sent_delim = r'(?:[":;.?!\(\)]|--)'
    token_delim = r'W+'
    join_pat = '\n'
    
    def __init__(self):
        pass
        
    def import_lines(self, src_file, start_line, end_line, strip=True):
        self.df0 = pd.DataFrame({'line':open(src_file,'r').readlines()})
        self.df0 = self.df0.loc[start_line:end_line]
        self.df0.index.name = 'line_id'
        if strip:
            self.df0['line'] = self.df0['line'].str.strip()
    
    def group_lines_by_milestone(self, div_name, div_pat, tmp_col='div_idx'):
        self.df0[div_name] = self.df0['line'].str.match(div_pat)
        self.df1 = self.df0.loc[self.df0[div_name], 'line'].copy()\
            .reset_index(drop=True).to_frame()\
            .rename(columns={'line':div_name})
        self.df1.index.name = div_name + '_id'
        self.df0[tmp_col] = None
        self.df0[tmp_col] = self.df0[self.df0[div_name]].apply(lambda x: x.index)
        self.df0[tmp_col] = self.df0[tmp_col].ffill()
        self.df0[tmp_col] = self.df0[tmp_col].astype('int')
        self.df1[div_name] = self.df0.groupby(tmp_col)['line'].apply(lambda x: self.join_pat.join(x[:]))\
            .to_frame().reset_index(drop=True)
        self.df1.index.name = div_name + '_id'
    
    def split_text_by_delimitter(self, div_name=None, div_pat=None, src_col=None):
        index_names = self.df1.index.names + [div_name + '_id']
        self.df1 = self.df1[src_col].str.split(div_pat, expand=True)\
            .stack().to_frame()\
            .rename(columns={0:div_name})
        self.df1.index.names = index_names
        self.df1[div_name] = self.df1[div_name].str.replace(self.join_pat, ' ')
        self.df1 = self.df1[~self.df1[div_name].str.match(r'^\s*$')]
    
    # Include POS recognization here since it is more efficient
    def tokenize_text(self, with_sent=True, normalize=True, add_pos=True):
        self.split_text_by_delimitter('sent', self.sent_delim, 'para')
        print('TK1', self.df1.head())
        self.split_text_by_delimitter('token', self.token_delim, 'sent')
        print('TK2', self.df1.head())
        if normalize:
            self.df1['token_norm'] = self.df1['token']\
                .str.lower().str.strip()
        if add_pos:
            idx = list(self.df1.index.names)[:-1]
            df = self.df1.groupby(idx).token.apply(lambda x: nltk.pos_tag(x.tolist()))\
                .apply(pd.Series).stack()\
                .to_frame().reset_index()\
                .rename(columns={'level_{}'.format(len(idx)):'token_id', 0:'pos'})\
                .set_index(idx + ['token_id'])
            self.df1['pos'] = df.pos.apply(lambda x: x[1])
        
#    def add_pos_to_tokens(self, idx=['chap_id','para_id','sent_id'], token_col='token'):
#        df = tokens.groupby(idx).token.apply(lambda x: nltk.pos_tag(x.tolist()))\
#            .apply(pd.Series).stack()\
#            .to_frame().reset_index()\
#            .rename(columns={'level_{}'.format(len(idx)):'token_id', 0:'pos'})\
#            .set_index(idx + ['token_id'])
#        tokens['pos'] = df.pos.apply(lambda x: x[1])
#        return tokens
#    
#    def create_vocab(df, col='token_norm'):
#        terms = df[col].value_counts()\
#            .to_frame().reset_index()\
#            .rename(columns={'index':'term',col:'n'})\
#            .sort_values('term').reset_index(drop=True)
#        terms.index.name = 'term_id'
#        terms['f'] = terms.n.div(terms.n.sum())
#        return terms
#    
#    def add_stems_to_vocab(df):
#        ps = PorterStemmer()
#        vocab['stem'] = vocab['term'].apply(lambda x: ps.stem(x))
#        return vocab
#    
#    def link_tokens_to_vocab(tokens, vocab, drop=False):
#        tokens['term_id'] = tokens['token_norm'].map(vocab.reset_index().set_index('term').term_id)
#        if drop:
#            del(tokens['token_norm'])
#        return tokens
#    
#    def identify_stopwords(vocab):
#        sw = set(stopwords.words('english'))
#        vocab['sw'] = vocab.term.apply(lambda x: 
#            x in sw or len(x) < 2 or not x.isalpha())
#        return vocab
#    
#    def remove_stopwords(df, vocab, term_id_col='term_id'):
#        df = df[df[term_id_col].isin(vocab[~vocab.sw].index.values)]
#        return df
#    
#    def create_doc_table(tokens, index=['chap_id', 'para_id']):
#        doc = tokens.groupby(index).term_id.count()\
#            .to_frame().rename(columns={'term_id':'n'})
#        return doc
#    
#    def create_bow(tokens, idx, index_name='doc_id'):
#        col = idx[-1]
#        bow = tokens.groupby(idx)[col].count()\
#            .to_frame().rename(columns={col:'n'})
#        if index_name:
#            bow.index.name = index_name
#        return bow
#    
#    def create_dtm(bow, fill_val=0):
#        dtm = bow.unstack().fillna(fill_val)
#        dtm.columns = dtm.columns.droplevel(0)
#        return dtm
#    
#    def compute_term_freq(dtm, vocab):
#        dtm_tf = dtm.apply(lambda x: x / x.sum(), 1)
#        vocab['tf_sum'] = dtm_tf.sum()
#        return dtm, vocab
#    
#    def compute_inv_doc_freq(dtm, vocab):
#        N = len(dtm.index)
#        dtm_idf = dtm.apply(lambda x: N / x[x > 0].count())
#        vocab['idf'] = dtm_idf
#        return dtm_idf, vocab
#    
#    def compute_tfidf(dtm, vocab, doc, bow, sw=False):
#        N = len(dtm.index)
#        dtm_tfidf = dtm.apply(lambda row: row / row.sum(), 1)\
#            .apply(lambda col: col * np.log(N/col[col > 0].count()))
#        vocab['tfidf_sum'] = dtm_tfidf.sum()
#        doc['tfidf_sum'] = dtm_tfidf.sum(1)
#        bow['tfidf'] = dtm_tfidf.stack().to_frame().rename(columns={0:'tfidf'})
#        return dtm_tfidf, vocab, doc, bow
#    
#    def compute_tfidh():
#        pass
#    
#    #def compute_tfidf2(dtmtf, vocab, bow, sw=False):
#    #    if sw:
#    #        V = vocab
#    #    else:
#    #        V = vocab[~vocab.sw]
#    #    dtm_tfidf = dtmtf.apply(lambda x: x * np.log(V.idf), 1)
#    #    #dtm_tfidf = dtmtf.multiply(np.log(V.idf), 1)
#    #    vocab['tfidf_sum1'] = dtm_tfidf.sum()
#    #    bow['tfidf1'] = dtm_tfidf.stack().to_frame().rename(columns={0:'tfidf'})
#    #    return dtm_tfidf, vocab, bow
#    
#    def get_term_id(vocab, term):
#        term_id = vocab[vocab.term==term].index[0] 
#        return term_id
#    
#    def get_term(vocab, term_id):
#        term = vocab.loc[term_id].term
#        return term
#    
#    # Put these in another class (use SA?)
#    
#    def put_to_db(db, df, table_name, index=True, if_exists='replace'):
#        r = df.to_sql(table_name, db, index=index, if_exists=if_exists)
#        return r
#        
#    def get_from_db(db, table_name):
#        df = pd.read_sql("SELECT * FROM {}".format(table_name), db)
#        return df
    

if __name__ == '__main__':
    
    corpus = Corpus()
    corpus.import_lines('../moby.txt', 318, 23238)
    corpus.group_lines_by_milestone('chap', r'^(?:ETYMOLOGY|CHAPTER \d+|Epilog)')
    corpus.split_text_by_delimitter('para', r'\n\n+', 'chap')
    corpus.tokenize_text()

    
#    import matplotlib.pyplot as plt
#    import seaborn as sns; sns.set()
#    
#    
#    pwd = '/Users/rca2t/Dropbox/Courses/DSI/DS5559/course-repo-od'
#    db = sqlite3.connect(pwd + '/moby.db')
#    
#    config = dict(
#        clips = dict(
#            src_file = pwd + '/moby.txt',
#            start_line = 318,
#            end_line = 23238            
#        ),
#        chap = dict(
#            div_name = 'chap',
#            div_pat = r'^(?:ETYMOLOGY|CHAPTER \d+|Epilog)',
#            src_idx = 'line_id',
#            src_col = 'line'
#        ),
#        para = dict(
#            div_name = 'para',
#            div_pat = r'\n\n+',
#            src_col = 'chap'
#        ),
#                
#        sent = dict(
#            div_name = 'sent',
#            div_pat = r'(?:[":;.?!\(\)]|--)',
#            src_col = 'para',
#            join_pat = ' '
#       ),  
#       token = dict(
#           div_name = 'token',
#           div_pat = r'\W+',
#           src_col = 'sent',
#           join_pat = ' '
#       )
#    )
#
#    print("SRC")
#    src = import_source(**config['clips'])
#    
#    print("CHAP")
#    chaps = group_by_milestone(src, **config['chap'])
#    
#    print("PARA")
#    paras = split_by_delimitter(chaps, **config['para'])
#    
#    print("SENT")
#    sents = split_by_delimitter(paras, **config['sent'])
#    
#    print("TOKENS")
#    tokens = split_by_delimitter(sents, **config['token'])
#    
#    print("VOCAB")
#    tokens = normalize_tokens(tokens)
#    tokens = add_pos_to_tokens(tokens)
#    vocab = create_vocab(tokens)
#    vocab = add_stems_to_vocab(vocab)
#    vocab = identify_stopwords(vocab)
#    tokens = link_tokens_to_vocab(tokens, vocab, drop=True)
##    tokens = remove_stopwords(tokens, vocab)
#    
#    print("DOC")
#    doc = create_doc_table(tokens, ['chap_id', 'para_id'])
#    
#    print("BOW")
#    bow = create_bow(tokens, ['chap_id', 'para_id', 'term_id'])
#
#    print("DTM")
#    dtm = create_dtm(bow)
#
#
#    print("TFIDF")
#    tfidf, vocab, doc, bow = compute_tfidf(dtm, vocab, doc, bow)
#    
##    print("TFIDF2")
##    dtmtf, vocab = compute_term_freq(dtm, vocab)
##    dtmidf, vocab = compute_inv_doc_freq(dtm, vocab) # DEPENDENCY ON DOC UNIT!
##    tfidf2, vocab, bow = compute_tfidf2(dtmidf, vocab, bow)
#    
#    print("PLOTS")
#    tfidf.sum(1).plot(figsize=(10,2))    
#    tfidf[[get_term_id(vocab, 'ahab'),get_term_id(vocab, 'whale')]].plot(figsize=(10,2))
#
##    put_to_db(db, f0, 'f0')
##    put_to_db(db, f1, 'f1_chap')
##    put_to_db(db, f2, 'f2_para')
##    put_to_db(db, f3, 'f3_sent')
##    put_to_db(db, tokens, 'docterm')
#    
