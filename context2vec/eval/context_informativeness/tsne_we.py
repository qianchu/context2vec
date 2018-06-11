
# coding: utf-8

# In[67]:


#!/usr/bin/env python
import numpy
import sys
import re
import nltk
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import pickle
    
from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader


if __name__ =="__main__":
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

    #preprocess word list
   

    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]

    word2freq_nostop={}
    index_filter=[]
    index_filter2index={}
    counter=0
    with open('word_freq') as f:
        for line in f:
            f_w=line.split()
            if len(f_w)>1:
                if f_w[1] in word2index and re.search('[a-zA-Z]',f_w[1])!=None and f_w[1] not in stopw :
                    word2freq_nostop[f_w[1]]=f_w[0]
                    index_filter.append(word2index[f_w[1]])
                    index_filter2index[word2index[f_w[1]]]=counter
                    counter+=1
    #fit tsne
    we_tsne=TSNE(n_components=2,verbose=5).fit_transform(w[index_filter,:])
    with open('we_tsne','w') as f:
        pickle.dump(we_tsne,f)
    with open('index_filter','w') as f_if:
        pickle.dump(index_filter, f_if)
    with open('index_filter2index','w') as f_if2f:
        pickle.dump(index_filter2index,f_if2f)

