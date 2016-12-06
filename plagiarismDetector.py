#! /usr/bin/python -tt
# -*- coding: utf-8 -*-

from __future__ import print_function  
from __future__ import division
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
import os
import string
import nltk
from nltk.probability import FreqDist
from nltk.util import ngrams
from collections import defaultdict
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
import numpy as np
import numpy.linalg as LA
import csv
from weka.classifiers import Classifier

c=""
title=""
tx=[]
a1_1=[]
b1_1=[]
c1_1=[]
a2_1=[]
b2_1=[]
c2_1=[]
a3_1=[]
b3_1=[]
c3_1=[]
a4_1=[]
b4_1=[]
c4_1=[]
a1_1_1=[]
b1_1_1=[]
c1_1_1=[]
a2_1_1=[]
b2_1_1=[]
c2_1_1=[]
a3_1_1=[]
b3_1_1=[]
c3_1_1=[]
a4_1_1=[]
b4_1_1=[]
c4_1_1=[]

t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
t10=[]
t11=[]
t12=[]
t13=[]
t14=[]
t15=[]
t16=[]
t17=[]
t18=[]
t19=[]
t20=[]
t21=[]
t22=[]
t23=[]
t24=[]
t25=[]
t26=[]
t27=[]
t28=[]
t29=[]
t30=[]
t31=[]
t32=[]
t33=[]
t34=[]
t35=[]
t36=[]
t37=[]
t38=[]
t39=[]
t40=[]
t41=[]
t42=[]
t43=[]
t44=[]
t45=[]
t46=[]
t47=[]
t48=[]
t49=[]
t50=[]
t51=[]
t52=[]
t53=[]
t54=[]
t55=[]
t56=[]
t57=[]
t58=[]
t59=[]
t60=[]
t61=[]
t62=[]
t63=[]
t64=[]
t65=[]
t66=[]
t67=[]
t68=[]
t69=[]
t70=[]
t71=[]
t72=[]
t73=[]
t74=[]
t75=[]
t76=[]
t77=[]
t78=[]
t79=[]
t80=[]
t81=[]
t82=[]
t83=[]
t84=[]
t85=[]
t86=[]
t87=[]
t88=[]
t89=[]
t90=[]
t91=[]
t92=[]
t93=[]
t94=[]
t95=[]
t96=[]
t97=[]
t98=[]
t99=[]
t100=[]
t101=[]
t102=[]
t103=[]
t104=[]
t105=[]
t106=[]
t107=[]
t108=[]
t109=[]
t110=[]
t111=[]
t112=[]
t113=[]
t114=[]
t115=[]
t116=[]
t117=[]
t118=[]
t119=[]
t120=[]

u1=[]
u2=[]
u3=[]
u4=[]
u5=[]
u6=[]
u7=[]
u8=[]
u9=[]
u10=[]
u11=[]
u12=[]
u13=[]
u14=[]
u15=[]
u16=[]
u17=[]
u18=[]
u19=[]
u20=[]
u21=[]
u22=[]
u23=[]
u24=[]
u25=[]
u26=[]
u27=[]
u28=[]
u29=[]
u30=[]
u31=[]
u32=[]
u33=[]
u34=[]
u35=[]
u36=[]
u37=[]
u38=[]
u39=[]
u40=[]
u41=[]
u42=[]
u43=[]
u44=[]
u45=[]
u46=[]
u47=[]
u48=[]
u49=[]
u50=[]
u51=[]
u52=[]
u53=[]
u54=[]
u55=[]
u56=[]
u57=[]
u58=[]
u59=[]
u60=[]
u61=[]
u62=[]
u63=[]
u64=[]
u65=[]
u66=[]
u67=[]
u68=[]
u69=[]
u70=[]
u71=[]
u72=[]
u73=[]
u74=[]
u75=[]
u76=[]
u77=[]
u78=[]
u79=[]
u80=[]
u81=[]
u82=[]
u83=[]
u84=[]
u85=[]
u86=[]
u87=[]
u88=[]
u89=[]
u90=[]
u91=[]
u92=[]
u93=[]
u94=[]
u95=[]
u96=[]
u97=[]
u98=[]
u99=[]
u100=[]
u101=[]
u102=[]
u103=[]
u104=[]
u105=[]
u106=[]
u107=[]
u108=[]
u109=[]
u110=[]
u111=[]
u112=[]
u113=[]
u114=[]
u115=[]
u116=[]
u117=[]
u118=[]
u119=[]
u120=[]

#sizet2=[]=3
#singles=[]

def preprocess(fnin, fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	buf = []
	id=""									
	for line in fin:
		#b=""
		#s=""
		if line.find(": ") > -1:
			#print (line)
			s=list(line)
			#print("hi "+s[0]+" "+s[1])
			s[0]=' '
			s[1]=' '
			#print("hii "+s[0]+" "+s[1])
			fout.write("\n")
			line=''.join(s)
			#line+=" "
			#print (line)
			
			b=' '.join(line.split())
			#print (b)
			fout.write("%s" % (b))
		else:
			b=' '.join(line.split())
			#print("check "+b)
			fout.write("%s" % (b))
		
		
	fin.close()
  	fout.close()


def make_lower(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		fout.write(line.lower())

def stemming_text_1(fnin,fnout):

	fout = open(fnout, 'wb')
	
	with open(fnin, 'rb') as f:
		stemmer = PorterStemmer() #problem from HERE
		for line in f:
			singles = []			
			for plural in line.split():
		       		singles.append(stemmer.stem(plural))
			fout.write(' '.join(singles)+'\n')

def lemmatizing_text_1(fnin,fnout):

	fout = open(fnout, 'wb')
	
	with open(fnin, 'rb') as f:
		lemmer = WordNetLemmatizer() #problem from HERE
		for line in f:
			singles = []			
			for plural in line.split():
		       		singles.append(lemmer.lemmatize(plural))
		
			#print (singles)			
			fout.write(' '.join(singles)+'\n')
			
def stop_words_punctuation_1(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.lower().translate(None, string.punctuation).split() 
			if len(word) >=4 and word not in stopwords.words('english')]), file=fout)

def punctuation_1(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.lower().translate(None, string.punctuation).split()]), file=fout)

def stop_words_1(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.lower().split() 
			if len(word) >=4 and word not in stopwords.words('english')]), file=fout)

def stop_words_punctuation_2(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.translate(None, string.punctuation).split() 
			if len(word) >=4 and word not in stopwords.words('english')]), file=fout)

def punctuation_2(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.translate(None, string.punctuation).split()]), file=fout)

def stop_words_2(fnin,fnout):

	fin = open(fnin, 'rb')
	#print fin
	fout = open(fnout, 'wb')
	
	#with open('input.txt','r') as inFile, open('output.txt','w') as outFile:
	for line in fin:
		print(' '.join([word for word in line.split() 
			if len(word) >=4 and word not in stopwords.words('english')]), file=fout)

def lexical_diversity(text):

	return len(text) / len(set(text))

def percentage(count, total):
	
	return 100 * count / total

def doc_compare(d1,d2):

	d1_k=set(d1.keys())
	d2_k=set(d2.keys())
	common=d1_k.intersection(d2_k)
	return common

def doc_summery1_1(ip,size):

	ip.seek(0,0)	
	global a1_1,b1_1,c1_1,c
	op = open('/home/saugata/Msc/nlp/corpus/Document_Summery/Document_Summery'+c+'_with_stopword_punctuation.txt', 'ab')
	#op.write("hi")
	infile = ip.readlines()

	i=0
	w1=[]
	for l in infile:
		i+=1			
		w1.extend(l.split())
	#print (i)
	#dict1={}
	
	for j in range(1,i+1):
		s="Document"
		s+=str(j)
		a1_1.append(s)
	b1_1=a1_1[:]
	#print (a1_1)	
	ip.seek(0,0)

	infile =ip.readlines()
	dict1=defaultdict(list)	
	i=1
	k=0
	
	for l in infile:
		ngram=ngrams(l.split(),size)
		#print (type(ngram))
		l1=(len(ngram))
		c1_1.append(l1)
		
		#print(w)
		#v1=lexical_diversity(l)
		
		voc = FreqDist(ngram)
		
		s="Document "+(str)(i)
		#print (len(l))
		#dict1[s].append(v1)
		#dict1[s].extend(voc)
		#op.write("hi")
		op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")

		pair1=str(size)+"gram list".ljust(40,' ')+"Count".ljust(20,' ')+"Word_list/Text".ljust(20,' ')+"Frequency".ljust(20,' ')
		op.write(pair1+'\n')		
		op.write("------------------------------------------------------------------------------------"+'\n')
		
		a1_1[k]=defaultdict(list)		

		for word in voc:
			#print(voc.freq(word))
			v2=percentage(voc[word],len(l))
			a1_1[k][word]=voc.freq(word)
	    		pair2 = str(word).ljust(40,' ')+ str(voc[word]).ljust(20,' ')+(str)(v2).ljust(20,' ')+str(voc.freq(word)).ljust(20,' ')
	    		op.write(pair2+'\n')
		i+=1
		k+=1
		op.write("------------------------------------------------------------------------------------"+'\n')
		op.write("Total "+str(size)+"grams: ".ljust(20,' '))
		op.write(str(l1).ljust(20,' ')+"\n\n\n")
	
	op.close()

def doc_summery2_1(ip,size):

	ip.seek(0,0)	
	global a2_1,b2_1,c2_1,c
	op = open('/home/saugata/Msc/nlp/corpus/Document_Summery/Document_Summery'+c+'_without_stopword_punctuation.txt', 'ab')
	#op.write("hi")
	infile = ip.readlines()

	i=0
	w1=[]
	for l in infile:
		i+=1			
		w1.extend(l.split())
	#print (i)
	#dict1={}
	
	for j in range(1,i+1):
		s="Document"
		s+=str(j)
		a2_1.append(s)
	b2_1=a2_1[:]
	#print (a1_1)	
	ip.seek(0,0)

	infile =ip.readlines()
	dict1=defaultdict(list)	
	i=1
	k=0
	
	for l in infile:
		ngram=ngrams(l.split(),size)
		l1=(len(ngram))
		c2_1.append(l1)
		
		#print(w)
		#v1=lexical_diversity(l)
		
		voc = FreqDist(ngram)
		
		s="Document "+(str)(i)
		#print (len(l))
		#dict1[s].append(v1)
		#dict1[s].extend(voc)
		#op.write("hi")
		op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")

		pair1=str(size)+"gram list".ljust(40,' ')+"Count".ljust(20,' ')+"Word_list/Text".ljust(20,' ')+"Frequency".ljust(20,' ')
		op.write(pair1+'\n')		
		op.write("------------------------------------------------------------------------------------"+'\n')
		
		a2_1[k]=defaultdict(list)		

		for word in voc:
			#print(voc.freq(word))
			v2=percentage(voc[word],len(l))
			a2_1[k][word]=voc.freq(word)
	    		pair2 = str(word).ljust(40,' ')+ str(voc[word]).ljust(20,' ')+(str)(v2).ljust(20,' ')+str(voc.freq(word)).ljust(20,' ')
	    		op.write(pair2+'\n')
		i+=1
		k+=1
		op.write("------------------------------------------------------------------------------------"+'\n')
		op.write("Total "+str(size)+"grams: ".ljust(20,' '))
		op.write(str(l1).ljust(20,' ')+"\n\n\n")
	
	op.close()

def doc_summery3_1(ip,size):

	ip.seek(0,0)	
	global a3_1,b3_1,c3_1,c
	op = open('/home/saugata/Msc/nlp/corpus/Document_Summery/Document_Summery'+c+'_without_stopword.txt', 'ab')
	#op.write("hi")
	infile = ip.readlines()

	i=0
	w1=[]
	for l in infile:
		i+=1			
		w1.extend(l.split())
	#print (i)
	#dict1={}
	
	for j in range(1,i+1):
		s="Document"
		s+=str(j)
		a3_1.append(s)
	b3_1=a3_1[:]
	#print (a1_1)	
	ip.seek(0,0)

	infile =ip.readlines()
	dict1=defaultdict(list)	
	i=1
	k=0
	
	for l in infile:
		ngram=ngrams(l.split(),size)
		l1=(len(ngram))
		c3_1.append(l1)
		
		#print(w)
		#v1=lexical_diversity(l)
		
		voc = FreqDist(ngram)
		
		s="Document "+(str)(i)
		#print (len(l))
		#dict1[s].append(v1)
		#dict1[s].extend(voc)
		#op.write("hi")
		op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")

		pair1=str(size)+"gram list".ljust(40,' ')+"Count".ljust(20,' ')+"Word_list/Text".ljust(20,' ')+"Frequency".ljust(20,' ')
		op.write(pair1+'\n')		
		op.write("------------------------------------------------------------------------------------"+'\n')
		
		a3_1[k]=defaultdict(list)		

		for word in voc:
			#print(voc.freq(word))
			v2=percentage(voc[word],len(l))
			a3_1[k][word]=voc.freq(word)
	    		pair2 = str(word).ljust(40,' ')+ str(voc[word]).ljust(20,' ')+(str)(v2).ljust(20,' ')+str(voc.freq(word)).ljust(20,' ')
	    		op.write(pair2+'\n')
		i+=1
		k+=1
		op.write("------------------------------------------------------------------------------------"+'\n')
		op.write("Total "+str(size)+"grams: ".ljust(20,' '))
		op.write(str(l1).ljust(20,' ')+"\n\n\n")
	
	op.close()

def doc_summery4_1(ip,size):

	ip.seek(0,0)	
	global a4_1,b4_1,c4_1,c
	op = open('/home/saugata/Msc/nlp/corpus/Document_Summery/Document_Summery'+c+'_without_punctuation.txt', 'ab')
	#op.write("hi")
	infile = ip.readlines()

	i=0
	w1=[]
	for l in infile:
		i+=1			
		w1.extend(l.split())
	#print (i)
	#dict1={}
	
	for j in range(1,i+1):
		s="Document"
		s+=str(j)
		a4_1.append(s)
	b4_1=a4_1[:]
	#print (a1_1)	
	ip.seek(0,0)

	infile =ip.readlines()
	dict1=defaultdict(list)	
	i=1
	k=0
	
	for l in infile:
		ngram=ngrams(l.split(),size)
		l1=(len(ngram))
		c4_1.append(l1)
		
		#print(w)
		#v1=lexical_diversity(l)
		
		voc = FreqDist(ngram)
		
		s="Document "+(str)(i)
		#print (len(l))
		#dict1[s].append(v1)
		#dict1[s].extend(voc)
		#op.write("hi")
		op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")

		pair1=str(size)+"gram list".ljust(40,' ')+"Count".ljust(20,' ')+"Word_list/Text".ljust(20,' ')+"Frequency".ljust(20,' ')
		op.write(pair1+'\n')		
		op.write("------------------------------------------------------------------------------------"+'\n')
		
		a4_1[k]=defaultdict(list)		

		for word in voc:
			#print(voc.freq(word))
			v2=percentage(voc[word],len(l))
			a4_1[k][word]=voc.freq(word)
	    		pair2 = str(word).ljust(40,' ')+ str(voc[word]).ljust(20,' ')+(str)(v2).ljust(20,' ')+str(voc.freq(word)).ljust(20,' ')
	    		op.write(pair2+'\n')
		i+=1
		k+=1
		op.write("------------------------------------------------------------------------------------"+'\n')
		op.write("Total "+str(size)+"grams: ".ljust(20,' '))
		op.write(str(l1).ljust(20,' ')+"\n\n\n")
	
	op.close()




def relative_frequency_model_1(size,lower,stemming):
	
	global a1_1,b1_1,c1_1,a2_1,b2_1,c2_1,a3_1,b3_1,c3_1,a4_1,b4_1,c4_1,c
	global t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30
	global t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60
	global t61,t62,t63,t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90
	global t91,t92,t93,t94,t95,t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111,t112,t113,t114,t115,t116,t117,t118,t119,t120
	
	op2 = open('/home/saugata/Msc/nlp/corpus/Frequency_Summary/Relative_Frequency_Summary'+c+'.txt', 'ab')
	op3 = open('/home/saugata/Msc/nlp/corpus/Frequency_Summary/Details_Frequency_Summary'+c+'.txt', 'ab')

	x=0
	q=[]
	max_percent=[]
	
	finding_max_matched_document=defaultdict(list)
	
	pair3="Percentage".rjust(60,' ')
	op2.write(pair3+'\n')		
	op2.write("-----------------------------------------------------------------------------------------------"+'\n')
	pair4="Stopword/Punctuation".rjust(50,' ')
	op2.write(pair4+'\n')		
	op2.write("-----------------------------------------------------------------------------------------------"+'\n')
	dict1=defaultdict(list)	
	dict2=defaultdict(list)	
	pair3="Word".ljust(30,' ')+"FrequencyA".ljust(40,' ')+"FrequencyB".ljust(20,' ')
	op2.write(pair3+'\n')
	for m in range(2,len(a1_1)):
		#print(m)
		common_words=doc_compare(a1_1[1],a1_1[m])
		p=len(common_words)
		q.append(p)
		finding_max_matched_document[p]=b1_1[m]
		#n=docu_length(a[m])
		#print (n)
		#print (str(a[m])+" "+str(len(a[m])))
		#print (common_words[0])
		#z=str(b[0])+" Vs. "+str(b[m])
		#op2.write(z+'\n')
		n1=0
		n2=0
		#if(len(common_words)>0):		
		for l in common_words:		

			pair4 = str(l).ljust(30,' ')+ str(a1_1[1][l]).ljust(40,' ')+str(a1_1[m][l]).ljust(20,' ')
			op2.write(pair4+'\n')
		op2.write("------------------------------------------------------------------------------------"+'\n')
		op2.write(str(b1_1[m]).ljust(15,' ')+"Matched words: ".ljust(15,' '))
		op2.write(str(p).ljust(10,' '))
		n=p/c1_1[m]
		dict1[round(n*100,2)]=b1_1[m]
		#print (dict1)
		dict2[b1_1[m]]=p
		#print ("hi")
		#print (dict2)
		max_percent.append(round(n*100,2))
		n1=max(max_percent)
		n2=min(max_percent)
		n3=round(n*100,2)

		if(size==1):		
			if(lower==1 and stemming==0):
				t1.append(n3)
			elif(lower==1 and stemming==1):		
				t2.append(n3)
			elif(lower==1 and stemming==2):
				t3.append(n3)
			elif(lower==0 and stemming==0):		
				t4.append(n3)	
			elif(lower==0 and stemming==1):
				t5.append(n3)
			elif(lower==0 and stemming==2):
				t6.append(n3)
		if(size==2):		
			if(lower==1 and stemming==0):
				t7.append(n3)
			elif(lower==1 and stemming==1):		
				t8.append(n3)
			elif(lower==1 and stemming==2):
				t9.append(n3)
			elif(lower==0 and stemming==0):		
				t10.append(n3)	
			elif(lower==0 and stemming==1):
				t11.append(n3)
			elif(lower==0 and stemming==2):
				t12.append(n3)

		if(size==3):		
			if(lower==1 and stemming==0):
				t13.append(n3)
			elif(lower==1 and stemming==1):		
				t14.append(n3)
			elif(lower==1 and stemming==2):
				t15.append(n3)
			elif(lower==0 and stemming==0):		
				t16.append(n3)	
			elif(lower==0 and stemming==1):
				t17.append(n3)
			elif(lower==0 and stemming==2):
				t18.append(n3)
		if(size==4):		
			if(lower==1 and stemming==0):
				t19.append(n3)
			elif(lower==1 and stemming==1):		
				t20.append(n3)
			elif(lower==1 and stemming==2):
				t21.append(n3)
			elif(lower==0 and stemming==0):		
				t22.append(n3)	
			elif(lower==0 and stemming==1):
				t23.append(n3)
			elif(lower==0 and stemming==2):
				t24.append(n3)
		if(size==5):		
			if(lower==1 and stemming==0):
				t25.append(n3)
			elif(lower==1 and stemming==1):		
				t26.append(n3)
			elif(lower==1 and stemming==2):
				t27.append(n3)
			elif(lower==0 and stemming==0):		
				t28.append(n3)	
			elif(lower==0 and stemming==1):
				t29.append(n3)
			elif(lower==0 and stemming==2):
				t30.append(n3)

		#print (n1)
		#print (dict1)
			#print (dict2)
		op2.write(str(n3)+'%'.ljust(5,' '))
		op2.write("------------------------------------------------------------------------------------"+'\n\n')
		#print (n)
		
		x+=1
		#print (finding_max_matched_document[18])
		#min_percent=max_percent[:]	
	
		#op2.write("------------------------------------------------------------------------------------"+'\n')
		#op2.write("Maximum Matched".ljust(20,' '))
	op2.write("------------------------------------------------------------------------------------"+'\n\n')	
	if(lower==1 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/Without Lemmatizing:  No match is found."+'\n')
	elif(lower==1 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==1 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/With Stop Words-Punctuation/Without Stemming/With Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/With Stop Words-Punctuation/Without Stemming/With Lemmating:  No match is found."+'\n')
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')	
	
	x=0
	q=[]
	max_percent=[]
	
	finding_max_matched_document=defaultdict(list)
	
	pair4="No Stopword/Punctuation".rjust(50,' ')
	op2.write(pair4+'\n')		
	op2.write("-----------------------------------------------------------------------------------------------"+'\n')
	dict1=defaultdict(list)	
	dict2=defaultdict(list)	
	
	for m in range(2,len(a2_1)):
		common_words=doc_compare(a2_1[1],a2_1[m])
		p=len(common_words)
		q.append(p)
		finding_max_matched_document[p]=b2_1[m]
		#n=docu_length(a[m])
		#print (n)
		#print (str(a[m])+" "+str(len(a[m])))
		#print (common_words[0])
		#z=str(b[0])+" Vs. "+str(b[m])
		#op2.write(z+'\n')
		n1=0
		n2=0
		#if(len(common_words)>0):		
		for l in common_words:		

			pair4 = str(l).ljust(30,' ')+ str(a2_1[1][l]).ljust(40,' ')+str(a2_1[m][l]).ljust(20,' ')
			op2.write(pair4+'\n')
		op2.write("------------------------------------------------------------------------------------"+'\n')
		op2.write(str(b2_1[m]).ljust(15,' ')+"Matched words: ".ljust(15,' '))
		op2.write(str(p).ljust(10,' '))
		n=p/c2_1[m]
		dict1[round(n*100,2)]=b2_1[m]
		#print (dict1)
		dict2[b2_1[m]]=p
		#print ("hi")
		#print (dict2)
		max_percent.append(round(n*100,2))
		n1=max(max_percent)
		n2=min(max_percent)
		n3=round(n*100,2)
		if(size==1):		
			if(lower==1 and stemming==0):
				t31.append(n3)
			elif(lower==1 and stemming==1):		
				t32.append(n3)
			elif(lower==1 and stemming==2):
				t33.append(n3)
			elif(lower==0 and stemming==0):		
				t34.append(n3)	
			elif(lower==0 and stemming==1):
				t35.append(n3)
			elif(lower==0 and stemming==2):
				t36.append(n3)

		if(size==2):		
			if(lower==1 and stemming==0):
				t37.append(n3)
			elif(lower==1 and stemming==1):		
				t38.append(n3)
			elif(lower==1 and stemming==2):
				t39.append(n3)
			elif(lower==0 and stemming==0):		
				t40.append(n3)	
			elif(lower==0 and stemming==1):
				t41.append(n3)
			elif(lower==0 and stemming==2):
				t42.append(n3)

		if(size==3):		
			if(lower==1 and stemming==0):
				t43.append(n3)
			elif(lower==1 and stemming==1):		
				t44.append(n3)
			elif(lower==1 and stemming==2):
				t45.append(n3)
			elif(lower==0 and stemming==0):		
				t46.append(n3)	
			elif(lower==0 and stemming==1):
				t47.append(n3)
			elif(lower==0 and stemming==2):
				t48.append(n3)
		if(size==4):		
			if(lower==1 and stemming==0):
				t49.append(n3)
			elif(lower==1 and stemming==1):		
				t50.append(n3)
			elif(lower==1 and stemming==2):
				t51.append(n3)
			elif(lower==0 and stemming==0):		
				t52.append(n3)	
			elif(lower==0 and stemming==1):
				t53.append(n3)
			elif(lower==0 and stemming==2):
				t54.append(n3)
		if(size==5):		
			if(lower==1 and stemming==0):
				t55.append(n3)
			elif(lower==1 and stemming==1):		
				t56.append(n3)
			elif(lower==1 and stemming==2):
				t57.append(n3)
			elif(lower==0 and stemming==0):		
				t58.append(n3)	
			elif(lower==0 and stemming==1):
				t59.append(n3)
			elif(lower==0 and stemming==2):
				t60.append(n3)

		
		
		#print (n1)
		#print (dict1)
			#print (dict2)
		op2.write(str(n3)+'%'.ljust(5,' '))
		op2.write("------------------------------------------------------------------------------------"+'\n\n')
			#print (n)
		
		x+=1
		#print (finding_max_matched_document[18])
		#min_percent=max_percent[:]	
	
		#op2.write("------------------------------------------------------------------------------------"+'\n')
		#op2.write("Maximum Matched".ljust(20,' '))
	op2.write("------------------------------------------------------------------------------------"+'\n\n')	
	if(lower==1 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmatizing:  No match is found."+'\n')
	elif(lower==1 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==1 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words-Punctuation/Without Stemming/With Lemmating:  No match is found."+'\n')
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')
	x=0
	q=[]
	max_percent=[]
	
	finding_max_matched_document=defaultdict(list)
	
	pair4="No Stopword".rjust(50,' ')
	op2.write(pair4+'\n')		
	op2.write("-----------------------------------------------------------------------------------------------"+'\n')
	dict1=defaultdict(list)	
	dict2=defaultdict(list)	
	pair3="Word".ljust(30,' ')+"FrequencyA".ljust(40,' ')+"FrequencyB".ljust(20,' ')
	op2.write(pair3+'\n')
	for m in range(2,len(a3_1)):
		common_words=doc_compare(a3_1[1],a3_1[m])
		p=len(common_words)
		q.append(p)
		finding_max_matched_document[p]=b3_1[m]
		#n=docu_length(a[m])
		#print (n)
		#print (str(a[m])+" "+str(len(a[m])))
		#print (common_words[0])
		#z=str(b[0])+" Vs. "+str(b[m])
		#op2.write(z+'\n')
		n1=0
		n2=0
		#if(len(common_words)>0):		
		for l in common_words:		

			pair4 = str(l).ljust(30,' ')+ str(a3_1[1][l]).ljust(40,' ')+str(a3_1[m][l]).ljust(20,' ')
			op2.write(pair4+'\n')
		op2.write("------------------------------------------------------------------------------------"+'\n')
		op2.write(str(b3_1[m]).ljust(15,' ')+"Matched words: ".ljust(15,' '))
		op2.write(str(p).ljust(10,' '))
		n=p/c3_1[m]
		dict1[round(n*100,2)]=b3_1[m]
		#print (dict1)
		dict2[b3_1[m]]=p
		#print ("hi")
		#print (dict2)
		max_percent.append(round(n*100,2))
		n1=max(max_percent)
		n2=min(max_percent)

		n3=round(n*100,2)
		if(size==1):		
			if(lower==1 and stemming==0):
				t61.append(n3)
			elif(lower==1 and stemming==1):		
				t62.append(n3)
			elif(lower==1 and stemming==2):
				t63.append(n3)
			elif(lower==0 and stemming==0):		
				t64.append(n3)	
			elif(lower==0 and stemming==1):
				t65.append(n3)
			elif(lower==0 and stemming==2):
				t66.append(n3)
		if(size==2):		
			if(lower==1 and stemming==0):
				t67.append(n3)
			elif(lower==1 and stemming==1):		
				t68.append(n3)
			elif(lower==1 and stemming==2):
				t69.append(n3)
			elif(lower==0 and stemming==0):		
				t70.append(n3)	
			elif(lower==0 and stemming==1):
				t71.append(n3)
			elif(lower==0 and stemming==2):
				t72.append(n3)

		if(size==3):		
			if(lower==1 and stemming==0):
				t73.append(n3)
			elif(lower==1 and stemming==1):		
				t74.append(n3)
			elif(lower==1 and stemming==2):
				t75.append(n3)
			elif(lower==0 and stemming==0):		
				t76.append(n3)	
			elif(lower==0 and stemming==1):
				t77.append(n3)
			elif(lower==0 and stemming==2):
				t78.append(n3)
		if(size==4):		
			if(lower==1 and stemming==0):
				t79.append(n3)
			elif(lower==1 and stemming==1):		
				t80.append(n3)
			elif(lower==1 and stemming==2):
				t81.append(n3)
			elif(lower==0 and stemming==0):		
				t82.append(n3)	
			elif(lower==0 and stemming==1):
				t83.append(n3)
			elif(lower==0 and stemming==2):
				t84.append(n3)
		if(size==5):		
			if(lower==1 and stemming==0):
				t85.append(n3)
			elif(lower==1 and stemming==1):		
				t86.append(n3)
			elif(lower==1 and stemming==2):
				t87.append(n3)
			elif(lower==0 and stemming==0):		
				t88.append(n3)	
			elif(lower==0 and stemming==1):
				t89.append(n3)
			elif(lower==0 and stemming==2):
				t90.append(n3)

		
		
		#print (n1)
		#print (dict1)
			#print (dict2)
		op2.write(str(n3)+'%'.ljust(5,' '))
		op2.write("------------------------------------------------------------------------------------"+'\n\n')
			#print (n)
		
		x+=1
		#print (finding_max_matched_document[18])
		#min_percent=max_percent[:]	
	
		#op2.write("------------------------------------------------------------------------------------"+'\n')
		#op2.write("Maximum Matched".ljust(20,' '))
	op2.write("------------------------------------------------------------------------------------"+'\n\n')	
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
	if(lower==1 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/Without Lemmatizing:  No match is found."+'\n')
	elif(lower==1 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==1 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Stop Words/Without Stemming/With Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Stop Words/Without Stemming/With Lemmating:  No match is found."+'\n')
	x=0
	q=[]
	max_percent=[]
	
	finding_max_matched_document=defaultdict(list)
	
	pair4="No Punctuation".rjust(50,' ')
	op2.write(pair4+'\n')		
	op2.write("-----------------------------------------------------------------------------------------------"+'\n')
	dict1=defaultdict(list)	
	dict2=defaultdict(list)	
	
	for m in range(2,len(a4_1)):
		common_words=doc_compare(a4_1[1],a4_1[m])
		p=len(common_words)
		q.append(p)
		finding_max_matched_document[p]=b4_1[m]
		#n=docu_length(a[m])
		#print (n)
		#print (str(a[m])+" "+str(len(a[m])))
		#print (common_words[0])
		#z=str(b[0])+" Vs. "+str(b[m])
		#op2.write(z+'\n')
		n1=0
		n2=0
		#if(len(common_words)>0):		
		for l in common_words:		

			pair4 = str(l).ljust(30,' ')+ str(a4_1[1][l]).ljust(40,' ')+str(a4_1[m][l]).ljust(20,' ')
			op2.write(pair4+'\n')
		op2.write("------------------------------------------------------------------------------------"+'\n')
		op2.write(str(b4_1[m]).ljust(15,' ')+"Matched words: ".ljust(15,' '))
		op2.write(str(p).ljust(10,' '))
		n=p/c4_1[m]
		dict1[round(n*100,2)]=b4_1[m]
		#print (dict1)
		dict2[b4_1[m]]=p
		#print ("hi")
		#print (dict2)
		max_percent.append(round(n*100,2))
		n1=max(max_percent)
		n2=min(max_percent)
		#print (n1)
		#print (dict1)
			#print (dict2)
		n3=round(n*100,2)

		if(size==1):		
			if(lower==1 and stemming==0):
				t91.append(n3)
			elif(lower==1 and stemming==1):		
				t92.append(n3)
			elif(lower==1 and stemming==2):
				t93.append(n3)
			elif(lower==0 and stemming==0):		
				t94.append(n3)	
			elif(lower==0 and stemming==1):
				t95.append(n3)
			elif(lower==0 and stemming==2):
				t96.append(n3)

		if(size==2):		
			if(lower==1 and stemming==0):
				t97.append(n3)
			elif(lower==1 and stemming==1):		
				t98.append(n3)
			elif(lower==1 and stemming==2):
				t99.append(n3)
			elif(lower==0 and stemming==0):		
				t100.append(n3)	
			elif(lower==0 and stemming==1):
				t101.append(n3)
			elif(lower==0 and stemming==2):
				t102.append(n3)

		if(size==3):		
			if(lower==1 and stemming==0):
				t103.append(n3)
			elif(lower==1 and stemming==1):		
				t104.append(n3)
			elif(lower==1 and stemming==2):
				t105.append(n3)
			elif(lower==0 and stemming==0):		
				t106.append(n3)	
			elif(lower==0 and stemming==1):
				t107.append(n3)
			elif(lower==0 and stemming==2):
				t108.append(n3)
		if(size==4):		
			if(lower==1 and stemming==0):
				t109.append(n3)
			elif(lower==1 and stemming==1):		
				t110.append(n3)
			elif(lower==1 and stemming==2):
				t111.append(n3)
			elif(lower==0 and stemming==0):		
				t112.append(n3)	
			elif(lower==0 and stemming==1):
				t113.append(n3)
			elif(lower==0 and stemming==2):
				t114.append(n3)
		if(size==5):		
			if(lower==1 and stemming==0):
				t115.append(n3)
			elif(lower==1 and stemming==1):		
				t116.append(n3)
			elif(lower==1 and stemming==2):
				t117.append(n3)
			elif(lower==0 and stemming==0):		
				t118.append(n3)	
			elif(lower==0 and stemming==1):
				t119.append(n3)
			elif(lower==0 and stemming==2):
				t120.append(n3)
		
		op2.write(str(n3)+'%'.ljust(5,' '))
		op2.write("------------------------------------------------------------------------------------"+'\n\n')
			#print (n)
		
		x+=1
		#print (finding_max_matched_document[18])
		#min_percent=max_percent[:]	
	
		#op2.write("------------------------------------------------------------------------------------"+'\n')
		#op2.write("Maximum Matched".ljust(20,' '))
	op2.write("------------------------------------------------------------------------------------"+'\n\n')	
	if(lower==1 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/Without Lemmatizing: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/Without Lemmatizing:  No match is found."+'\n')
	elif(lower==1 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==1 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Lowercase/Without Punctuations/Without Stemming/With Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==0):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==1):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/With Stemming/Without Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/With Stemming/Without Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/With Stemming/Without Lemmating:  No match is found."+'\n')
	elif(lower==0 and stemming==2):	
		if(n1>0):	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/With Lemmating: "+str(dict2[dict1[n1]])+'('+str(n1)+'%)'+" finds in "+dict1[n1]+" which is maximum.\n")	
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/With Lemmating: "+str(dict2[dict1[n2]])+'('+str(n2)+'%)'+" finds in "+dict1[n2]+" which is minimum.\n")
		else:
			op3.write(str(size)+"gram/Uppercase/Without Punctuations/Without Stemming/With Lemmating:  No match is found."+'\n')	
	
	op3.write("------------------------------------------------------------------------------------"+'\n\n')
				
	op2.close()
	op3.close()


def jac_summery1_1(ip,size,lower,stemming):

	ip.seek(0,0)	
	global u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,c

	op = open('/home/saugata/Msc/nlp/corpus/Jaccard_Similarity/Similarity_Summery'+c+'_with_stopword_punctuation.txt', 'ab')
	
	infile = ip.readlines()
	i=0
	p=0
	for l in infile:	
		p+=1
		if p>1:	
			a=set(ngrams(l.split(),size))
			break

	ip.seek(0,0)
	
	for l in infile:
		i+=1
		if i>2:
			n3=0
			op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")
			b=set(ngrams(l.split(),size))
			sim=jaccardian_similarity(a,b)
			n3=round(sim,2)
			s="Jaccard Similarity".ljust(40,' ')
			op.write(s)
			s=str(n3).rjust(70,' ')
			op.write(s+'\n')
			op.write("------------------------------------------------------------------------------------"+'\n')
			print (str(i)+"->"+str(sim))	
			if(size==1):
				if(lower==1 and stemming==0):
					u1.append(n3)
				elif(lower==1 and stemming==1):		
					u2.append(n3)
				elif(lower==1 and stemming==2):
					u3.append(n3)
				elif(lower==0 and stemming==0):		
					u4.append(n3)	
				elif(lower==0 and stemming==1):
					u5.append(n3)
				elif(lower==0 and stemming==2):
					u6.append(n3)
			if(size==2):		
				if(lower==1 and stemming==0):
					u7.append(n3)
				elif(lower==1 and stemming==1):		
					u8.append(n3)
				elif(lower==1 and stemming==2):
					u9.append(n3)
				elif(lower==0 and stemming==0):		
					u10.append(n3)	
				elif(lower==0 and stemming==1):
					u11.append(n3)
				elif(lower==0 and stemming==2):
					u12.append(n3)
			if(size==3):		
				if(lower==1 and stemming==0):
					u13.append(n3)
				elif(lower==1 and stemming==1):		
					u14.append(n3)
				elif(lower==1 and stemming==2):
					u15.append(n3)
				elif(lower==0 and stemming==0):		
					u16.append(n3)	
				elif(lower==0 and stemming==1):
					u17.append(n3)
				elif(lower==0 and stemming==2):
					u18.append(n3)
			if(size==4):		
				if(lower==1 and stemming==0):
					u19.append(n3)
				elif(lower==1 and stemming==1):		
					u20.append(n3)
				elif(lower==1 and stemming==2):
					u21.append(n3)
				elif(lower==0 and stemming==0):		
					u22.append(n3)	
				elif(lower==0 and stemming==1):
					u23.append(n3)
				elif(lower==0 and stemming==2):
					u24.append(n3)
			if(size==5):		
				if(lower==1 and stemming==0):
					u25.append(n3)
				elif(lower==1 and stemming==1):		
					u26.append(n3)
				elif(lower==1 and stemming==2):
					u27.append(n3)
				elif(lower==0 and stemming==0):		
					u28.append(n3)	
				elif(lower==0 and stemming==1):
					u29.append(n3)
				elif(lower==0 and stemming==2):
					u30.append(n3)

					
	op.close()	

def jac_summery2_1(ip,size,lower,stemming):

	ip.seek(0,0)	
	global u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45,u46,u47,u48,u49,u50,u51,u52,u53,u54,u55,u56,u57,u58,u59,u60,c
	
	op = open('/home/saugata/Msc/nlp/corpus/Jaccard_Similarity/Similarity_Summery'+c+'_without_stopword_punctuation.txt', 'ab')
	
	infile = ip.readlines()
	i=0
	p=0
	for l in infile:	
		p+=1
		if p>1:	
			a=set(ngrams(l.split(),size))
			break

	ip.seek(0,0)
	
	for l in infile:
		i+=1
		if i>2:
			n3=0
			op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")
			b=set(ngrams(l.split(),size))
			sim=jaccardian_similarity(a,b)
			n3=round(sim,2)
			s="Jaccard Similarity".ljust(40,' ')
			op.write(s)
			s=str(n3).rjust(70,' ')
			op.write(s+'\n')
			op.write("------------------------------------------------------------------------------------"+'\n')
			print (str(i)+"->"+str(sim))
			if(size==1):		
				if(lower==1 and stemming==0):
					u31.append(n3)
				elif(lower==1 and stemming==1):		
					u32.append(n3)
				elif(lower==1 and stemming==2):
					u33.append(n3)
				elif(lower==0 and stemming==0):		
					u34.append(n3)	
				elif(lower==0 and stemming==1):
					u35.append(n3)
				elif(lower==0 and stemming==2):
					u36.append(n3)
			if(size==2):		
				if(lower==1 and stemming==0):
					u37.append(n3)
				elif(lower==1 and stemming==1):		
					u38.append(n3)
				elif(lower==1 and stemming==2):
					u39.append(n3)
				elif(lower==0 and stemming==0):		
					u40.append(n3)	
				elif(lower==0 and stemming==1):
					u41.append(n3)
				elif(lower==0 and stemming==2):
					u42.append(n3)
			if(size==3):		
				if(lower==1 and stemming==0):
					u43.append(n3)
				elif(lower==1 and stemming==1):		
					u44.append(n3)
				elif(lower==1 and stemming==2):
					u45.append(n3)
				elif(lower==0 and stemming==0):		
					u46.append(n3)	
				elif(lower==0 and stemming==1):
					u47.append(n3)
				elif(lower==0 and stemming==2):
					u48.append(n3)
			if(size==4):		
				if(lower==1 and stemming==0):
					u49.append(n3)
				elif(lower==1 and stemming==1):		
					u50.append(n3)
				elif(lower==1 and stemming==2):
					u51.append(n3)
				elif(lower==0 and stemming==0):		
					u52.append(n3)	
				elif(lower==0 and stemming==1):
					u53.append(n3)
				elif(lower==0 and stemming==2):
					u54.append(n3)
			if(size==5):		
				if(lower==1 and stemming==0):
					u55.append(n3)
				elif(lower==1 and stemming==1):		
					u56.append(n3)
				elif(lower==1 and stemming==2):
					u57.append(n3)
				elif(lower==0 and stemming==0):		
					u58.append(n3)	
				elif(lower==0 and stemming==1):
					u59.append(n3)
				elif(lower==0 and stemming==2):
					u60.append(n3)

			
	op.close()		

def jac_summery3_1(ip,size,lower,stemming):

	ip.seek(0,0)	
	global u61,u62,u63,u64,u65,u66,u67,u68,u69,u70,u71,u72,u73,u74,u75,u76,u77,u78,u79,u80,u81,u82,u83,u84,u85,u86,u87,u88,u89,u90,c
	
	op = open('/home/saugata/Msc/nlp/corpus/Jaccard_Similarity/Similarity_Summery'+c+'_without_stopword.txt', 'ab')
	
	infile = ip.readlines()
	i=0
	p=0
	for l in infile:	
		p+=1
		if p>1:	
			a=set(ngrams(l.split(),size))
			break
	ip.seek(0,0)
	
	for l in infile:
		i+=1
		if i>2:  
			n3=0
			op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")
			b=set(ngrams(l.split(),size))
			sim=jaccardian_similarity(a,b)
			n3=round(sim,2)
			s="Jaccard Similarity".ljust(40,' ')
			op.write(s)
			s=str(n3).rjust(70,' ')
			op.write(s+'\n')
			op.write("------------------------------------------------------------------------------------"+'\n')
			print (str(i)+"->"+str(sim))	
			if(size==1):		
				if(lower==1 and stemming==0):
					u61.append(n3)
				elif(lower==1 and stemming==1):		
					u62.append(n3)
				elif(lower==1 and stemming==2):
					u63.append(n3)
				elif(lower==0 and stemming==0):		
					u64.append(n3)	
				elif(lower==0 and stemming==1):
					u65.append(n3)
				elif(lower==0 and stemming==2):
					u66.append(n3)
			if(size==2):		
				if(lower==1 and stemming==0):
					u67.append(n3)
				elif(lower==1 and stemming==1):		
					u68.append(n3)
				elif(lower==1 and stemming==2):
					u69.append(n3)
				elif(lower==0 and stemming==0):		
					u70.append(n3)	
				elif(lower==0 and stemming==1):
					u71.append(n3)
				elif(lower==0 and stemming==2):
					u72.append(n3)
			if(size==3):		
				if(lower==1 and stemming==0):
					u73.append(n3)
				elif(lower==1 and stemming==1):		
					u74.append(n3)
				elif(lower==1 and stemming==2):
					u75.append(n3)
				elif(lower==0 and stemming==0):		
					u76.append(n3)	
				elif(lower==0 and stemming==1):
					u77.append(n3)
				elif(lower==0 and stemming==2):
					u78.append(n3)
			if(size==4):		
				if(lower==1 and stemming==0):
					u79.append(n3)
				elif(lower==1 and stemming==1):		
					u80.append(n3)
				elif(lower==1 and stemming==2):
					u81.append(n3)
				elif(lower==0 and stemming==0):		
					u82.append(n3)	
				elif(lower==0 and stemming==1):
					u83.append(n3)
				elif(lower==0 and stemming==2):
					u84.append(n3)
			if(size==5):		
				if(lower==1 and stemming==0):
					u85.append(n3)
				elif(lower==1 and stemming==1):		
					u86.append(n3)
				elif(lower==1 and stemming==2):
					u87.append(n3)
				elif(lower==0 and stemming==0):		
					u88.append(n3)	
				elif(lower==0 and stemming==1):
					u89.append(n3)
				elif(lower==0 and stemming==2):
					u90.append(n3)

							
	op.close()	

def jac_summery4_1(ip,size,lower,stemming):

	ip.seek(0,0)	
	global u91,u92,u93,u94,u95,u96,u97,u98,u99,u100,u101,u102,u103,u104,u105,u106,u107,u108,u109,u110,u111,u112,u113,u114,u115,u116,u117,u118,u119,u120,c
	
	op = open('/home/saugata/Msc/nlp/corpus/Jaccard_Similarity/Similarity_Summery'+c+'_without_punctuation.txt', 'ab')
	
	infile = ip.readlines()
	i=0
	p=0
	for l in infile:	
		p+=1
		if p>1:	
			a=set(ngrams(l.split(),size))
			break

	ip.seek(0,0)
	
	for l in infile:
		i+=1
		if i>2:
			n3=0
			op.write("---------------------------------Document "+ (str)(i)+"---------------------------------"+"\n")
			b=set(ngrams(l.split(),size))
			sim=jaccardian_similarity(a,b)
			n3=round(sim,2)
			s="Jaccard Similarity".ljust(40,' ')
			op.write(s)
			s=str(n3).rjust(70,' ')
			op.write(s+'\n')
			op.write("------------------------------------------------------------------------------------"+'\n')
			print (str(i)+"->"+str(sim))		
			if(size==1):		
				if(lower==1 and stemming==0):
					u91.append(n3)
				elif(lower==1 and stemming==1):		
					u92.append(n3)
				elif(lower==1 and stemming==2):
					u93.append(n3)
				elif(lower==0 and stemming==0):		
					u94.append(n3)	
				elif(lower==0 and stemming==1):
					u95.append(n3)
				elif(lower==0 and stemming==2):
					u96.append(n3)
			if(size==2):		
				if(lower==1 and stemming==0):
					u97.append(n3)
				elif(lower==1 and stemming==1):		
					u98.append(n3)
				elif(lower==1 and stemming==2):
					u99.append(n3)
				elif(lower==0 and stemming==0):		
					u100.append(n3)	
				elif(lower==0 and stemming==1):
					u101.append(n3)
				elif(lower==0 and stemming==2):
					u102.append(n3)
			if(size==3):		
				if(lower==1 and stemming==0):
					u103.append(n3)
				elif(lower==1 and stemming==1):		
					u104.append(n3)
				elif(lower==1 and stemming==2):
					u105.append(n3)
				elif(lower==0 and stemming==0):		
					u106.append(n3)	
				elif(lower==0 and stemming==1):
					u107.append(n3)
				elif(lower==0 and stemming==2):
					u108.append(n3)
			if(size==4):		
				if(lower==1 and stemming==0):
					u109.append(n3)
				elif(lower==1 and stemming==1):		
					u110.append(n3)
				elif(lower==1 and stemming==2):
					u111.append(n3)
				elif(lower==0 and stemming==0):		
					u112.append(n3)	
				elif(lower==0 and stemming==1):
					u113.append(n3)
				elif(lower==0 and stemming==2):
					u114.append(n3)
			if(size==5):		
				if(lower==1 and stemming==0):
					u115.append(n3)
				elif(lower==1 and stemming==1):		
					u116.append(n3)
				elif(lower==1 and stemming==2):
					u117.append(n3)
				elif(lower==0 and stemming==0):		
					u118.append(n3)	
				elif(lower==0 and stemming==1):
					u119.append(n3)
				elif(lower==0 and stemming==2):
					u120.append(n3)
					
	op.close()

def jaccardian_similarity(a,b):

	similarity=float(len(a.intersection(b))*1.0/len(a.union(b)))
	return similarity

def convert(n):
	content = []
	name = ''
	global title
	if n.endswith('.csv') == True:
		name = n.replace('.csv', '')

	#print 'Opening CSV file.'     
	with open(n, 'rb') as csvfile:
		lines = csv.reader(csvfile, delimiter = ',')
        	for row in lines:
               		content.append(row)
        csvfile.close()
       	
	#print 'Converting to ARFF file.\n'
        title = str(name) + '.arff'
        new_file = open(title, 'w')

        new_file.write('@relation ' + str(name)+ '\n\n')

        #get attribute type input
        for i in range(len(content[0])-1):
            #attribute_type = raw_input('Is the type of ' + str(content[0][i]) + ' numeric or nominal? ')
	    attribute_type="numeric"
            new_file.write('@attribute \'' + str(content[0][i]) + '\' ' + str(attribute_type) + '\n')

        #create list for class attribute
        last = len(content[0])
        class_items = []
        for i in range(len(content)):
            name = content[i][last-1]
            if name not in class_items:
                class_items.append(content[i][last-1])
            else:
                pass  
        del class_items[0]
    
        string = '{' + ','.join(sorted(class_items)) + '}'
	string1 = '{HR,LR,NC,NP}'
        new_file.write('@attribute ' + str(content[0][last-1]) + ' ' + str(string1) + '\n')

        #write data
        new_file.write('\n@data\n')

        del content[0]
        for row in content:
            new_file.write(','.join(row) + '\n')

        #close file
        new_file.close()
        #sleep(2)

def lcs(a, b):
    #print (a,b)	
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return result

def splitParagraphIntoSentences(i):

	sentenceEnder=re.compile('[.!?]')
	sentenceList=sentenceEnder.split(i)
	return sentenceList

def common(id1):
	
	ipp= open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt", 'r')
	op = open('/home/saugata/Msc/nlp/corpus/Prediction/Common Chunks/Common.txt', 'ab')
	
	ipp.seek(0,0)
	infile = ipp.readlines()
	
	k=0
	w=[]
	#global a1_1,b1_1,c1_1

	i=0
	for l in infile:
		i+=1			
	#print i
	for j in range(1,i+1):
		s="Document"
		s+=str(j)
	#	a1_1.append(s)
	#b1_1=a1_1[:]

	dict1=defaultdict(list)	
	ref=[]
	z=0
	for f in infile:
		z+=1
		if z>1:
			ref.extend(splitParagraphIntoSentences(f))
			break	
	#print ref
	i=0
	len1=[]
	w2=0
	for l in infile:
		i+=1
		cand=[]
		if i==id1:
			w2=l.split()
			s="Document "+(str)(id1)
			op.write("---------------------------------Document "+ (str)(i-1)+"---------------------------------"+"\n")
			pair1="LCS".ljust(40,' ')
			op.write(pair1+'\n')		
			op.write("------------------------------------------------------------------------------------"+'\n')
					
			#print l[0:10]
			cand.extend(splitParagraphIntoSentences(l))
			#print cand
			if(cand[len(cand)-1]==' \n'):
				del cand[len(cand)-1]
			#x=0
			#a1_1[k]=defaultdict(list)
			#print ref[x]
			#print len(cand)
			lcs1=''
			sum1=0
			
			lcs1=lcs(ref,cand)
			pair2 = str(lcs1).ljust(40,' ')
		    	op.write(pair2+'\n')
			w=lcs1.split()	
			op.write("------------------------------------------------------------------------------------"+'\n')
			op.write("Total Matched chunks: ".ljust(15,' '))
			op.write(str(len(w)).ljust(10,' ')+"out of "+str(len(w2))+" words\n")
			#op.write(str(n3)+'%'.ljust(5,' '))
			op.write("------------------------------------------------------------------------------------"+'\n\n')
	#print "Lengths "+len1,	
	#print len(len1)	
	op.close()
	ipp.close()


def predict(n,x,m):
	
	global title
	predictions=None
	c1=None	
	j=2
	op = open('/home/saugata/Msc/nlp/corpus/Prediction/Predictions.txt', 'ab')	
	op.write("---------------------------------"+m+"---------------------------------"+"\n")
	run = convert(n)
	c1 = Classifier(name='weka.classifiers.trees.J48')
	c1.train(x)
	predictions = c1.predict(title)
	#print (predictions)
	for i in predictions:
		j+=1
		print (i.predicted)
		if i.predicted<>'NP':
			common(j)
		op.write((str)(i)+"\n")
	op.write("------------------------------------------------------------------------------------"+'\n')
	op.write("------------------------------------------------------------------------------------"+'\n')
	op.close()

def main():

	global a1_1,b1_1,c1_1,a2_1,b2_1,c2_1,a3_1,b3_1,c3_1,a4_1,b4_1,c4_1
	global a1_1_1,b1_1_1,c1_1_1
	global a2_1_1,b2_1_1,c2_1_1
	global a3_1_1,b3_1_1,c3_1_1
	global a4_1_1,b4_1_1,c4_1_1
	global c,tx
	global title
	for i in range(1000):
		tx.append("?");

	global t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30
	global t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60
	global t61,t62,t63,t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90
	global t91,t92,t93,t94,t95,t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111,t112,t113,t114,t115,t116,t117,t118,t119,t120
	global u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30
	global u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45,u46,u47,u48,u49,u50,u51,u52,u53,u54,u55,u56,u57,u58,u59,u60
	global u61,u62,u63,u64,u65,u66,u67,u68,u69,u70,u71,u72,u73,u74,u75,u76,u77,u78,u79,u80,u81,u82,u83,u84,u85,u86,u87,u88,u89,u90
	global u91,u92,u93,u94,u95,u96,u97,u98,u99,u100,u101,u102,u103,u104,u105,u106,u107,u108,u109,u110,u111,u112,u113,u114,u115,u116,u117,u118,u119,u120

	header=[(
"Stop Words/Punctuations1(L)","Stop Words/Punctuations/Stemming1(L)","Stop Words/Punctuations/Lammetizing1(L)","Stop Words/Punctuations1(NC)","Stop Words/Punctuations/Stemming1(NC)","Stop Words/Punctuations/Lammetizing1(NC)",
"Stop Words/Punctuations2(L)","Stop Words/Punctuations/Stemming2(L)","Stop Words/Punctuations/Lammetizing2(L)","Stop Words/Punctuations2(NC)","Stop Words/Punctuations/Stemming2(NC)","Stop Words/Punctuations/Lammetizing2(NC)",
"Stop Words/Punctuations3(L)","Stop Words/Punctuations/Stemming3(L)","Stop Words/Punctuations/Lammetizing3(L)","Stop Words/Punctuations3(NC)","Stop Words/Punctuations/Stemming3(NC)","Stop Words/Punctuations/Lammetizing3(NC)",
"Stop Words/Punctuations4(L)","Stop Words/Punctuations/Stemming4(L)","Stop Words/Punctuations/Lammetizing4(L)","Stop Words/Punctuations4(NC)","Stop Words/Punctuations/Stemming4(NC)","Stop Words/Punctuations/Lammetizing4(NC)",
"Stop Words/Punctuations5(L)","Stop Words/Punctuations/Stemming5(L)","Stop Words/Punctuations/Lammetizing5(L)","Stop Words/Punctuations5(NC)","Stop Words/Punctuations/Stemming5(NC)","Stop Words/Punctuations/Lammetizing5(NC)",
"(No Stop Words/Punctuations)1(L)","(No Stop Words/Punctuations)/Stemming1(L)","(No Stop Words/Punctuations)/Lammetizing1(L)","(No Stop Words/Punctuations)1(NC)","(No Stop Words/Punctuations)/Stemming1(NC)","(No Stop Words/Punctuations)/Lammetizing1(NC)",
"(No Stop Words/Punctuations)2(L)","(No Stop Words/Punctuations)/Stemming2(L)","(No Stop Words/Punctuations)/Lammetizing2(L)","(No Stop Words/Punctuations)2(NC)","(No Stop Words/Punctuations)/Stemming2(NC)","(No Stop Words/Punctuations)/Lammetizing2(NC)",
"(No Stop Words/Punctuations)3(L)","(No Stop Words/Punctuations)/Stemming3(L)","(No Stop Words/Punctuations)/Lammetizing3(L)","(No Stop Words/Punctuations)3(NC)","(No Stop Words/Punctuations)/Stemming3(NC)","(No Stop Words/Punctuations)/Lammetizing3(NC)",
"(No Stop Words/Punctuations)4(L)","(No Stop Words/Punctuations)/Stemming4(L)","(No Stop Words/Punctuations)/Lammetizing4(L)","(No Stop Words/Punctuations)4(NC)","(No Stop Words/Punctuations)/Stemming4(NC)","(No Stop Words/Punctuations)/Lammetizing4(NC)",
"(No Stop Words/Punctuations)5(L)","(No Stop Words/Punctuations)/Stemming5(L)","(No Stop Words/Punctuations)/Lammetizing5(L)","(No Stop Words/Punctuation)5(NC)","(No Stop Words/Punctuations)/Stemming5(NC)","(No Stop Words/Punctuations)/Lammetizing5(NC)",
"No Stop Words1(L)","No Stop Words/Stemming1(L)","No Stop Words/Lammetizing1(L)","No Stop Words1(NC)","No Stop Words/Stemming1(NC)","No Stop Words/Lammetizing1(NC)",
"No Stop Words2(L)","No Stop Words/Stemming2(L)","No Stop Words/Lammetizing2(L)","No Stop Words2(NC)","No Stop Words/Stemming2(NC)","No Stop Words/Lammetizing2(NC)",
"No Stop Words3(L)","No Stop Words/Stemming3(L)","No Stop Words/Lammetizing3(L)","No Stop Words3(NC)","No Stop Words/Stemming3(NC)","No Stop Words/Lammetizing3(NC)",
"No Stop Words4(L)","No Stop Words/Stemming4(L)","No Stop Words/Lammetizing4(L)","No Stop Words4(NC)","No Stop Words/Stemming4(NC)","No Stop Words/Lammetizing4(NC)",
"No Stop Words5(L)","No Stop Words/Stemming5(L)","No Stop Words/Lammetizing5(L)","No Stop Words5(NC)","No Stop Words/Stemming5(NC)","No Stop Words/Lammetizing5(NC)",
"No Punctuations1(L)","No Punctuations/Stemming1(L)","No Punctuations/Lammetizing1(L)","No Punctuations1(NC)","No Punctuations/Stemming1(NC)","No Punctuations/Lammetizing1(NC)",
"No Punctuations2(L)","No Punctuations/Stemming2(L)","No Punctuations/Lammetizing2(L)","No Punctuations2(NC)","No Punctuations/Stemming2(NC)","No Punctuations/Lammetizing2(NC)",
"No Punctuations3(L)","No Punctuations/Stemming3(L)","No Punctuations/Lammetizing3(L)","No Punctuations3(NC)","No Punctuations/Stemming3(NC)","No Punctuations/Lammetizing3(NC)",
"No Punctuations4(L)","No Punctuations/Stemming4(L)","No Punctuations/Lammetizing4(L)","No Punctuations4(NC)","No Punctuations/Stemming4(NC)","No Punctuations/Lammetizing4(NC)",
"No Punctuations5(L)","No Punctuations/Stemming5(L)","No Punctuations/Lammetizing5(L)","No Punctuations5(NC)","No Punctuations/Stemming5(NC)","No Punctuations/Lammetizing5(NC)")]
	
	header1=[(
"Stop Words/Punctuation1(L)","Stop Words /Punctuations/Stemming1(L)","Stop Words /Punctuations/Lammetizing1(L)","Stop Words/Punctuations1(NC)","Stop Words/Punctuations/Stemming1(NC)","Stop Words/Punctuations/Lammetizing1(NC)","(No Stop Words/Punctuation)1(L)","(No Stop Words/Punctuations)/Stemming1(L)","(No Stop Words/Punctuations)/Lammetizing1(L)","(No Stop Words/Punctuations)1(NC)","(No Stop Words/Punctuations)/Stemming1(NC)","(No Stop Words/Punctuations)/Lammetzing1(NC)","(No Stop Words/Punctuation)2(L)","(No Stop Words/Punctuations)/Stemming2(L)","(No Stop Words/Punctuations)/Lammetizing2(L)","(No Stop Words/Punctuations)/Stemming2(NC)","(No Stop Words/Punctuations)/Lammetzing2(NC)","No Stopwords1(L)","No Stop Words/Stemming1(L)","No Stop Words/Lammetizing1(L)","No Stop Words1(NC)","No Stop Words/Stemming1(NC)","No Stop Words/Lammetizing1(NC)","No Punctuations/Stemming1(L)","No Punctuations/Lammetizing1(L)","No Punctuations/ Lammetizing1(NC)","Class")]
	
	header2=[(
"Stop Words/Punctuation1(L)","Stop Words /Punctuations/Stemming1(L)","Stop Words /Punctuations/Lammetizing1(L)","Stop Words/Punctuations1(NC)","Stop Words/Punctuations/Stemming1(NC)","Stop Words/Punctuations/Lammetizing1(NC)","(No Stop Words/Punctuation)1(L)","(No Stop Words/Punctuations)/Stemming1(L)","(No Stop Words/Punctuations)/Lammetizing1(L)","(No Stop Words/Punctuations)1(NC)","(No Stop Words/Punctuations)/Stemming1(NC)","(No Stop Words/Punctuations)/Lammetzing1(NC)","No Stopwords1(L)","No Stop Words/Stemming1(L)","No Stop Words/Lammetizing1(L)","No Stop Words1(NC)","No Stop Words/Stemming1(NC)","No Stop Words/Lammetizing1(NC)","No Punctuations1(L)","No Punctuations/Stemming1(L)","No Punctuations/Lammetizing1(L)","No Punctuations1(NC)","No Punctuations/ Stemming1(NC)","No Punctuations/ Lammetizing1(NC)","No Punctuations2(L)","Class")]

	d1='/home/saugata/Msc/nlp/corpus/Preprocessed Files'
	d2='/home/saugata/Msc/nlp/corpus/Data'
	d3='/home/saugata/Msc/nlp/corpus/Document_Summery'
	d4='/home/saugata/Msc/nlp/corpus/Frequency_Summary'
	d5='/home/saugata/Msc/nlp/corpus/Jaccard_Similarity'
	d6='/home/saugata/Msc/nlp/corpus/Prediction'
	d7='/home/saugata/Msc/nlp/corpus/Prediction/Common Chunks'

	os.makedirs(d1)
	os.makedirs(d2)
	os.makedirs(d3)
	os.makedirs(d4)
	os.makedirs(d5)
	os.makedirs(d6)
	os.makedirs(d7)
	#op4 = open('trainE.csv', 'ab')
	#wr=csv.writer(op4,quoting=csv.QUOTE_ALL)
	
	#wr.writerow()
	
	lemmatizing_stemming=0
	#op3 = open('Details_Frequency_Summary.txt', 'ab')
	c=raw_input("Enter File Name: ")
		
	preprocess(c+".txt", '/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt")
	make_lower('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x0.txt")
	stop_words_punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x1.txt")
	stop_words_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x2.txt")
	punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x3.txt")

	ip1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x0.txt", 'r')
	ip2 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x1.txt", 'r')
	ip3 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x2.txt", 'r')
	ip4 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x3.txt", 'r')

	for i in range(1,6):
		lower1=1
		lemmatizing_stemming=0
		doc_summery1_1(ip1,i)#with stop/punctuation	
		doc_summery2_1(ip2,i)
		doc_summery3_1(ip3,i)#witout stop words only
		doc_summery4_1(ip4,i)#witout punctuations only
		
		jac_summery1_1(ip1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
	
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
	stop_words_punctuation_2('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x1_1.txt")
	stop_words_2('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x2_1.txt")
	punctuation_2('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x3_1.txt")

	ip1_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt", 'r')
	ip2_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x1_1.txt", 'r')
	ip3_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x2_1.txt", 'r')
	ip4_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x3_1.txt", 'r')
	

	for i in range(1,6):
		lower1=0
		lemmatizing_stemming=0
		doc_summery1_1(ip1_1,i)#with stop/punctuation	
		doc_summery2_1(ip2_1,i)
		doc_summery3_1(ip3_1,i)#witout stop words only
		doc_summery4_1(ip4_1,i)#witout punctuations only
		
		jac_summery1_1(ip1_1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2_1,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3_1,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4_1,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
	stemming_text_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt")
	make_lower('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_0.txt")
	stop_words_punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_1.txt")
	stop_words_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_2.txt")
	punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_3.txt")
		
	ip1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_0.txt", 'r')
	ip2 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_1.txt", 'r')
	ip3 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_2.txt", 'r')
	ip4 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_3.txt", 'r')
	
	for i in range(1,6):
		lower1=1
		lemmatizing_stemming=1
		doc_summery1_1(ip1,i)#with stop/punctuation	
		doc_summery2_1(ip2,i)
		doc_summery3_1(ip3,i)#witout stop words only
		doc_summery4_1(ip4,i)#witout punctuations only
		
		jac_summery1_1(ip1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
	stop_words_punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_1_1.txt")
	stop_words_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_2_2.txt")
	punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_3_3.txt")
		
	ip1_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4.txt", 'r')
	ip2_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_1_1.txt", 'r')
	ip3_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_2_2.txt", 'r')
	ip4_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x4_3_3.txt", 'r')
	
	for i in range(1,6):
		lower1=0
		lemmatizing_stemming=1
		
		doc_summery1_1(ip1_1,i)#with stop/punctuation	
		doc_summery2_1(ip2_1,i)
		doc_summery3_1(ip3_1,i)#witout stop words only
		doc_summery4_1(ip4_1,i)#witout punctuations only
		
		jac_summery1_1(ip1_1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2_1,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3_1,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4_1,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
	#op3.write("------------------------------------------------------------------------------------"+'\n\n')
	lemmatizing_text_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt")	
	make_lower('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_0.txt")
	stop_words_punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_1.txt")
	stop_words_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_2.txt")
	punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_0.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_3.txt")
		
	ip1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_0.txt", 'r')
	ip2 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_1.txt", 'r')
	ip3 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_2.txt", 'r')
	ip4 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_3.txt", 'r')
	

	for i in range(1,6):
		lower1=1
		lemmatizing_stemming=2
		doc_summery1_1(ip1,i)#with stop/punctuation	
		doc_summery2_1(ip2,i)
		doc_summery3_1(ip3,i)#witout stop words only
		doc_summery4_1(ip4,i)#witout punctuations only
		
		jac_summery1_1(ip1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
	
	stop_words_punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_1_1.txt")
	stop_words_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_2_2.txt")
	punctuation_1('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt",'/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_3_3.txt")
		
	ip1_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5.txt", 'r')
	ip2_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_1_1.txt", 'r')
	ip3_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_2_2.txt", 'r')
	ip4_1 = open('/home/saugata/Msc/nlp/corpus/Preprocessed Files/'+c+"x5_3_3.txt", 'r')
	
	for i in range(1,6):
		lower1=0
		lemmatizing_stemming=2
		doc_summery1_1(ip1_1,i)#with stop/punctuation	
		doc_summery2_1(ip2_1,i)
		doc_summery3_1(ip3_1,i)#witout stop words only
		doc_summery4_1(ip4_1,i)#witout punctuations only
		
		jac_summery1_1(ip1_1,i,lower1,lemmatizing_stemming)#with stop/punctuation	
		jac_summery2_1(ip2_1,i,lower1,lemmatizing_stemming)
		jac_summery3_1(ip3_1,i,lower1,lemmatizing_stemming)#witout stop words only
		jac_summery4_1(ip4_1,i,lower1,lemmatizing_stemming)#witout punctuations only
		relative_frequency_model_1(i,lower1,lemmatizing_stemming)
		#op3.write("------------------------------------------------------------------------------------"+'\n\n')
		
		a1_1=[]
		b1_1=[]
		c1_1=[]
		a2_1=[]
		b2_1=[]
		c2_1=[]
		a3_1=[]
		b3_1=[]
		c3_1=[]
		a4_1=[]
		b4_1=[]
		c4_1=[]
		
	

	#print (singles)	
	#print("hi")
	#print (' '.join(singles))	
	#print (a[0])
	#print (c)	
	
	#n=l3/l2
	#print(n)
	#print(x) 
	
	#for i in a:	
	#	k=i.keys()
	#	v=i.values()
	#	for k,v in i.items():
	#		#print ("hi")		
	#		print (k,v)
	#	print('\n\n')
		
	#preprocess("corpus.txt", "x.txt")
	#make_lower("x.txt","x0.txt")
	#stop_words_punctuation_1("x0.txt","x1.txt")
	#stop_words_1("x0.txt","x2.txt")
	#punctuation_1("x0.txt","x3.txt")

	print("list: ")
	#print(t1)
	#for i in range()
	with open('/home/saugata/Msc/nlp/corpus/Data/'+c+'.csv','wb') as out:
		csv.writer(out,delimiter=',',quoting=csv.QUOTE_MINIMAL).writerows(header+zip(t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,t61,t62,t63,t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90
,t91,t92,t93,t94,t95,t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111,t112,t113,t114,t115,t116,t117,t118,t119,t120
))
	with open('/home/saugata/Msc/nlp/corpus/Data/'+c+'_filtered.csv','wb') as out:
		csv.writer(out,delimiter=',',quoting=csv.QUOTE_MINIMAL).writerows(header1+zip(t1,t2,t3,t4,t5,t6,t31,t32,t33,t34,t35,t36,t37,t38,t39,t41,t42,t61,t62,t63,t64,t65,t66,t92,t93,t96,tx))

	n='/home/saugata/Msc/nlp/corpus/Data/'+c+'_filtered.csv'
	x='filtered_train_1.arff'
	m="Frequency Comparison"
	predict(n,x,m)
	
	with open('/home/saugata/Msc/nlp/corpus/Data/'+'Jaccard_Similarity_Score_'+c+'.csv','wb') as out:
		csv.writer(out,delimiter=',',quoting=csv.QUOTE_MINIMAL).writerows(header+zip(u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45,u46,u47,u48,u49,u50,u51,u52,u53,u54,u55,u56,u57,u58,u59,u60,u61,u62,u63,u64,u65,u66,u67,u68,u69,u70,u71,u72,u73,u74,u75,u76,u77,u78,u79,u80,u81,u82,u83,u84,u85,u86,u87,u88,u89,u90,u91,u92,u93,u94,u95,u96,u97,u98,u99,u100,u101,u102,u103,u104,u105,u106,u107,u108,u109,u110,u111,u112,u113,u114,u115,u116,u117,u118,u119,u120
))
	with open('/home/saugata/Msc/nlp/corpus/Data/'+'Jaccard_Similarity_Score_'+c+'_filtered.csv','wb') as out:
		csv.writer(out,delimiter=',',quoting=csv.QUOTE_MINIMAL).writerows(header2+zip(u1,u2,u3,u4,u5,u6,u32,u33,u34,u35,u36,u37,u62,u63,u64,u65,u66,u67,u92,u93,u94,u95,u96,u97,u98,tx))

	n='/home/saugata/Msc/nlp/corpus/Data/'+'Jaccard_Similarity_Score_'+c+'_filtered.csv'
	x='filtered_train_j1.arff'
	m="Jaccard Similarity"
	predict(n,x,m)
	#run = convert(n)
	#c2 = Classifier(name='weka.classifiers.lazy.IBk', ckargs={'-K':1})
	#c2.train('filtered_train_j1.arff')
	#predictionss = c2.predict(title)
	#for i in predictionss:
	#	print (i)
	ip1.close()
	ip2.close()
	ip3.close()
	ip4.close()
	ip1_1.close()
	ip2_1.close()
	ip3_1.close()
	ip4_1.close()
	#op3.close()

	
if __name__ == "__main__":
	
	main()
