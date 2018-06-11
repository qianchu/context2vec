
# coding: utf-8

# In[1]:


#!/usr/bin/env python
import numpy
import six
import sys
import traceback
import re
import pickle


from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader
import sklearn


# In[2]:


def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity)


# In[3]:



if __name__=="__main__":
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        model_param_file='../../../model_dir/context2vec.ukwac.model.params'
    else:
        if len(sys.argv) < 2:
            print >> sys.stderr, "Usage: %s <model-param-file>"  % (sys.argv[0])
            sys.exit(1)

            model_param_file = sys.argv[1]
            

    #read in model
    model_reader = ModelReader(model_param_file)
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model


# In[12]:


# #PCA
# from sklearn.manifold import TSNE
# we_tsne=TSNE(n_components=2,verbose=5).fit_transform(w)
we_tsne=pickle.load(open('we_tsne'))
index_filter2index=pickle.load(open('index_filter2index'))
index_filter=pickle.load(open('index_filter'))


# In[224]:


# setup            
n_result = 20  # number of search result to show
gpu = -1 # todo: make this work with gpu

if gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()    
xp = cuda.cupy if gpu >= 0 else numpy



# In[246]:


# test we_pc
# s = numpy.sqrt((we_pc * we_pc).sum(1))
# s[s==0.] = 1.
# we_pc /= s.reshape((s.shape[0], 1))
target_v=w_filter[index2index_filter[word2index['quasar']]]
similarity = (w_filter.dot(target_v)+1.0)/2
top_words_i=[]

#test most similar words

count = 0
for i in (-similarity).argsort():
            if xp.isnan(similarity[i]) or similarity[i]==1.0:
                continue
            print('{0}: {1}'.format(index2word[index_filter[i]], similarity[i]))
            count += 1
            top_words_i.append(i)
            if count == n_result:
                break
                
top_vec=w_filter[top_words_i,:]
sum(sum(top_vec.dot(top_vec.T)))/(n_result**2)


# In[268]:


#a test sentence
#sentence="We 're getting food like CANBIM beef milk and flour and dividing into packages and taking into inaccessible places ."
#sentence="It would probably be more fun than listening to some self-deceiving gilgul spin her miserable COTTAN ."
#sentence="It describes Aspergillas flavus which is a fungus mainly found in drought stressed SHAIN in the South-Eastern United States ."
sentence="Arrange the lettuce leaves over a large serving platter with the tomatoes, TROULT , radishes and spring onions."
words=sentence.split()
context_embed= model_reader.model.context2vec(words, words.index('TROULT'))
print (xp.sqrt((context_embed * context_embed).sum()))
context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
# similarity 
similarity = (w_filter.dot(context_embed)+1.0)/2
top_words_i=[]
count = 0
for i in (-similarity).argsort():
            if xp.isnan(similarity[i]):
                continue
            print('{0}: {1}'.format(index2word[index_filter[i]], similarity[i]))
            count += 1
            top_words_i.append(i)
            if count == n_result:
                break

top_vec=w_filter[top_words_i,:]
sum(sum(top_vec.dot(top_vec.T)))/(n_result**2)


# In[221]:





# In[ ]:


#graph

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib notebook')
top_n=50
top_n_sort=(-similarity).argsort()[:top_n]
x,y=zip(*we_pc[top_n_sort,:])

c=zip(similarity[top_n_sort],[0.2]*top_n,[0.4]*top_n)
plt.scatter(x, y,c=c,alpha=0.5)


#annotate text
for cur_i,i in enumerate(top_n_sort):
    plt.annotate(index2word[i],(x[cur_i],y[cur_i]))
plt.ylim(-0.2,0.2)
plt.xlim(-0.2,0.2)
plt.show()
#c


