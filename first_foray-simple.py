
# coding: utf-8

# <a href="https://colab.research.google.com/github/ontoligent-design/DS5559/blob/master/first_foray.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Assumptions
# 
# * Do not preserve punction and whitespace 
# * Work with a single text

# # Settings

#%% Test

foo = 'bar'

#%% Configs
WIDE = (15,3)
THIN = (5,15)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def source_to_tokens(src_file, start_line=0, end_line=None):
    """Transform plaintext source file into F1 """
    text = open(src_file,'r').readlines()
    text = pd.DataFrame(text, columns=['line'])
    text = text.index.name = 'line_id'
    text['line'] = text['line'].str.strip()
    text = text.loc[start_line:end_line].copy()
    text = text.reset_index(drop=True)
    text.index.name = 'line_id'
    return text

# Chapters

moby_lines['chap_ms'] = moby_lines.line  .str.match(r'^(?:ETYMOLOGY|CHAPTER \d+|Epilog)')
moby_lines['chap_idx'] = None # Not sure why this has to be initialized
moby_lines['chap_idx'] = moby_lines[moby_lines.chap_ms]  .apply(lambda x: x.index)
moby_lines.chap_idx = moby_lines.chap_idx.ffill()
moby_lines.chap_idx = moby_lines['chap_idx'].astype('int')
moby_chaps = moby_lines.groupby('chap_idx').line  .apply(lambda x: '\n'.join(x[:])).to_frame().reset_index(drop=True)
moby_chaps.index.name = 'chap_id'

# Paragraphs

moby_paras = moby_chaps.line.str.split(r'\n\n+', expand=True).stack().to_frame().reset_index().rename(columns={'level_1':'para_id', 0:'para'})
moby_paras.set_index(['chap_id','para_id'], inplace=True)
moby_paras.para = moby_paras.para.str.replace(r'\n', ' ')


# Sentences
moby_sent = moby_paras.para.str.split(r'(?:[":;.?!\(\)]|--)', expand=True)  .stack()  .to_frame().reset_index().rename(columns={'level_2':'sent_id', 0:'sent'})
moby_sent.set_index(['chap_id', 'para_id', 'sent_id'], inplace=True)
moby_sent = moby_sent[~moby_sent.sent.str.match(r'^\s*$')]
moby_sent['new_sent_idx'] = moby_sent.groupby(['chap_id','para_id']).cumcount()
moby_sent = moby_sent.reset_index()
moby_sent.rename(columns={'sent_id':'delete_me'}, inplace=True)
moby_sent.rename(columns={'new_sent_idx':'sent_id'}, inplace=True)
moby_sent = moby_sent.set_index(['chap_id','para_id','sent_id'])['sent']  .to_frame()

# Tokens
moby_tokens = moby_sent.sent.str.split(r'\W+', expand=True).stack()  .to_frame().reset_index().rename(columns={'level_3':'token_id', 0:'token'})
moby_tokens.set_index(['chap_id', 'para_id', 'sent_id', 'token_id'], 
                      inplace=True)



moby_tokens['norm'] = moby_tokens.token.str.lower()


# In[28]:


moby_tokens.head()


# # F2: Create Vocabulary

# In[29]:


moby_vocab = pd.DataFrame(moby_tokens.token.str.lower().sort_values().unique(), columns=['term'])
moby_vocab.index.name = 'term_id'


# In[30]:


moby_vocab.sample(10)


# ## Get Term ID function

# In[31]:


def term_id(term):
    try:
        return moby_vocab.query("term == @term").index[0]
    except IndexError as e:
        return None


# ## Stopwords
# 
# We get our stopwords from an external source. We could have used NLTK.

# In[32]:


stopwords = requests  .get('https://algs4.cs.princeton.edu/35applications/stopwords.txt')  .text.split('\n')


# In[33]:


stopwords[:10]


# In[34]:


moby_vocab['sw'] = moby_vocab.term.apply(lambda x: 
                                         x in stopwords 
                                         or len(x) < 2 
                                         or not x.isalpha())


# In[35]:


moby_vocab[moby_vocab.term.str.match(r'\d(?:nd|th|st|rd)')]


# ## Replace terms in  with IDs

# In[36]:


moby_tokens['term_id'] = moby_tokens.norm.map(moby_vocab.reset_index()  .set_index('term').term_id)  .fillna(-1).astype('int')


# In[37]:


moby_tokens.head()


# In[38]:


moby_tokens = moby_tokens[['token','term_id']]


# In[39]:


moby_tokens.head()


# ## Add raw term counts to vocab

# In[40]:


moby_vocab['n'] = moby_tokens.groupby('term_id').term_id.count()


# In[41]:


n_words = moby_vocab.n.sum()
n_words_adj = moby_vocab[~moby_vocab.sw].n.sum()


# In[42]:


n_words, n_words_adj


# In[43]:


moby_vocab['freq'] = moby_vocab.n.div(n_words)


# In[44]:


moby_vocab.loc[~moby_vocab.sw, 'adj_freq'] = moby_vocab[~moby_vocab.sw].n  .div(n_words_adj)


# In[45]:


moby_vocab.sample(10)


# In[46]:


moby_vocab.n.plot(figsize=WIDE)


# In[47]:


moby_vocab.freq.plot(figsize=WIDE)


# In[48]:


moby_vocab.adj_freq.plot(figsize=WIDE)


# In[49]:


moby_vocab.plot(kind='scatter', x='freq', y='adj_freq')


# In[50]:


moby_vocab.loc[moby_vocab.adj_freq.idxmax()].term


# In[51]:


moby_vocab.describe()


# In[52]:


moby_vocab = moby_vocab.query('sw == 0 & n > 2') 


# In[53]:


moby_vocab.describe()


# ## Create lite version of tokens

# In[54]:


moby_tokens = moby_tokens[moby_tokens.term_id.isin(moby_vocab.index.values)]


# In[55]:


moby_tokens


# ## Graph top terms

# In[56]:


top_terms = moby_vocab.sort_values('n', ascending=False).head(30)
plt.figure(figsize=(5,10))
sns.barplot(data=top_terms, y='term', x='n', orient='h')
plt.show()


# # Dispersion Plots

# ## Convert tokens into OHE matrix

# In[57]:


kahuna = pd.get_dummies(moby_tokens.reset_index()['term_id']).T
kahuna['term'] = moby_vocab.term
kahuna = kahuna.reset_index().set_index('term').drop('index', axis=1)


# In[58]:


kahuna.head()


# In[59]:


terms = ['stubb', 'ahab','whale', 'starbuck', 'queequeg', 'ishmael', 'white', 'sea', 'ship', 'church', 'death']


# In[60]:


viz_df = kahuna.loc[terms].T  .stack()  .to_frame()  .reset_index()  .rename(columns={'level_0': 't', 'level_1':'term', 0:'n'})


# In[61]:


viz_df[viz_df.n > 0].sample(5)


# In[91]:


plt.figure(figsize=(15,5))
sns.stripplot(y='term', x='t', data=viz_df[viz_df.n == 1],
 orient='h', marker="$|$", color='navy', size=15, jitter=0)
plt.show()


# # F3: Create BOW

# ## BOW by Chap

# In[63]:


moby_bow_chaps = moby_tokens  .groupby(['chap_id','term_id'])  .term_id.count()  .to_frame().rename(columns={'term_id':'n'})


# In[64]:


moby_bow_chaps.head()


# # F3: Create DTM

# ## DTM by Chap

# In[65]:


moby_dtm_chaps = moby_bow_chaps.unstack().fillna(0)
moby_dtm_chaps.columns = moby_dtm_chaps.columns.droplevel()


# In[66]:


moby_dtm_chaps.head()


# # F3: Create TFIDF Matrix

# ## Get N docs

# In[67]:


N = len(moby_dtm_chaps.index)


# In[68]:


N


# ## TFIDF by Chap

# In[69]:


moby_dtm_tfidf_chaps = moby_dtm_chaps.apply(lambda row: row / row.sum(), 1).apply(lambda col: col * np.log(N/col[col > 0].count()))


# In[70]:


moby_bow_chaps['tfidf'] = moby_dtm_tfidf_chaps.stack().to_frame().rename(columns={0:'tfidf'})


# In[71]:


moby_dtm_tfidf_chaps.head()


# # Term Frequency Graphs

# ## TFIDF by Chap

# In[89]:


moby_dtm_tfidf_chaps.T.loc[term_id('ahab')].plot(figsize=WIDE, title='ahab')
plt.show()


# In[88]:


moby_dtm_tfidf_chaps.T.loc[term_id('whale')].plot(figsize=WIDE, title='whale')
plt.show()


# In[90]:


moby_dtm_tfidf_chaps.T.loc[term_id('whale')].plot(figsize=WIDE, legend=True)
moby_dtm_tfidf_chaps.T.loc[term_id('whales')].plot(figsize=WIDE, legend=True)


# # Term Correlations

# In[74]:


corr_terms = [term_id(term) for term in terms]
corr_cols = {term_id(term):term for term in terms}


# In[75]:


def corr_plot_terms(terms, dtm, title='Foo'):
    plt.figure(figsize = (10,10))
    print(title)
    corr = dtm[corr_terms].rename(columns=corr_cols).corr()
    sns.heatmap(corr, vmax=.3, annot=True, center=0, 
              cmap='RdYlGn',
              square=True, linewidths=.5, 
              cbar_kws={"shrink": .5})
    plt.show()


# In[76]:


def corr_frame_terms(terms, dtm, title='Foo'):
  print(title)
  corr = dtm[corr_terms].rename(columns=corr_cols).corr()
  corr.index.name = 'src_term'
  corr = corr.stack().to_frame()    .reset_index()    .rename(columns={0:'corr'})    .sort_values('corr').reset_index()
  corr = corr.query('src_term != term_id').copy()
  corr['test'] = corr.index % 2
  corr = corr[corr.test == 0]
  corr = corr[['src_term','term_id','corr']]
  corr.columns = ['src_term','dst_term','corr']
  return pd.concat([corr.head(), corr.tail()])    .sort_values('corr', ascending=False)


# In[77]:


corr_frame_terms(terms, moby_dtm_chaps, 'By Chap')


# In[78]:


corr_frame_terms(terms, moby_dtm_tfidf_chaps, 'By Chap')


# ## By Chap

# In[80]:


corr_plot_terms(terms, moby_dtm_chaps, 'By Chap')


# In[230]:


corr_plot_terms(terms, moby_dtm_tfidf_chaps, 'By Chap')


# # PMI

# In[164]:


# n_tokens = moby_bow.n.sum().sum()
# n_tokens

# moby_dtm_chaps.sum().div(n_tokens).sort_values(ascending=False).head()

# moby_vocab.loc[~moby_vocab.sw, 'p'] = moby_dtm_chaps.sum().div(n_tokens)

# moby_vocab[~moby_vocab.sw].head()

# sns.scatterplot(data=moby_vocab, x='adj_freq', y='p')
# plt.show()


