from __future__ import division
import string
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np
import itertools as it
from porter2stemmer import Porter2Stemmer
import math


tokenize = lambda doc: doc.lower().split(" ")
#LabelDocs = []
#MainDocuments_List=[]

# read file from corpus.txt and store the documents in a list form separated on the bases of #
HashedDocuments_List = []
HashIndex_DocName = []
def file_read_from_corpus(filename): #test corpus endsem
	name = 1
	with open(filename,'r') as f:
		for key,group in it.groupby(f,lambda line: line.startswith('#')):
        		if not key:
            			group = list(group)
            			string = ""
				for line in group:
					line = line.rstrip('\n')
					string = string + line
				HashedDocuments_List.append(string)
				HashIndex_DocName.append(('doc_'+str(name)))
				name+=1


def file_read(filename,doc_list):  #test set (midsem corpus)
	document = []
	with open(filename) as f:
    		document = f.read()
		document = unicode(document,errors='ignore')
	MainDocuments_List.append(document)



def read_S08():
	for i in range(1,5):	#test set (midsem corpus)
		for j in range(1,11):
			string = "S08/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			string1 = "S08-" + str(i) + "-a" + str(j)
			LabelDocs.append(string1)
			file_read(string,MainDocuments_List)

def read_S09():
	for i in range(1,6):	#test set (midsem corpus)
		for j in range(1,11):
			string = "S09/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			string1 = "S09-" + str(i) + "-a" + str(j)
			LabelDocs.append(string1)
			file_read(string,MainDocuments_List)


def read_S10():		#test set (midsem corpus)
	for i in range(1,7):
		for j in range(1,11):
			string = "S10/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			string1 = "S10-" + str(i) + "-a" + str(j)
			LabelDocs.append(string1)
			file_read(string,MainDocuments_List)

def tf_matrix(HashedDocuments_List): # Calucale term document matrix
	term_document_matrix = CountVectorizer(tokenizer=tokenize)
	term_doc_representation = term_document_matrix.fit_transform(HashedDocuments_List)
	#print term_doc_representation
	return term_doc_representation

def tfidf(HashedDocuments_List,QueryString): # Calculate tfidf matrix and query vector
	tfidf_matrix= TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
	tfidf_rep = tfidf_matrix.fit_transform(HashedDocuments_List)
	query_response = tfidf_matrix.transform([QueryString])
	return tfidf_matrix,tfidf_rep,query_response

def bm25_function(filled_col_ind,tfidf_rep,term_doc_rep): ####	
	k=1 #given
	b=0.5
	resulting_doc_score=[]
	
	# BM 25 Formula implementation
	for i in range(0,term_doc_rep.get_shape()[0]):
		sum=0
		for index in filled_col_ind:
			sum+=(tfidf_rep[i,index]*1.0*(k+1))/(term_doc_rep[i,index]+k*(1-b+(b*term_doc_rep[i,:].sum(axis=None))/((term_doc_rep[:,:].sum(axis=None))*1.0/term_doc_rep.get_shape()[0])))
		resulting_doc_score.append(sum)

	return resulting_doc_score




def bm25_result_print(filled_col_ind,tfidf_rep,term_doc_rep): ####
	
	result=bm25_function(filled_col_ind,tfidf_rep,term_doc_rep)
	result_indices = sorted(range(len(result)), key=lambda k: result[k])[-10:]
	result_indices.reverse()

	for index in (result_indices):
		print("BM25: doc_name,result",HashIndex_DocName[index],result[index],"Index:",index)


def bm25_function(filled_col_ind,tfidf_rep,term_doc_rep): # final working function
	final_scores = {}
	total_docs = term_doc_rep.get_shape()[0]
	avg_doc_length = term_doc_rep[:,:].sum(axis = None) 	
	avg_doc_length = float(avg_doc_length)/total_docs
	word_doc_freq = [] #No of documents containing that word
	word_freq = [] #No of times word appears in each document
	word_doc_position = [] #Doc numbers containing that word

	for i in filled_col_ind:
		number_of_docs = 0 #count
		word_frequency = []
		word_document_position = []
		for j in range(0,total_docs):
			if(term_doc_rep[j,i]>0):
				number_of_docs+=1
				word_frequency.append(term_doc_rep[j,i])
				word_document_position.append(j)				
				final_scores[j]=0
				
		word_doc_freq.append(number_of_docs)
		word_freq.append(word_frequency)
		word_doc_position.append(word_document_position)
	
	#print len(word_doc_freq)
	#print len(word_freq)
	#print len(word_doc_position)
	
	for i in range(0,len(filled_col_ind)):
		term_freq = word_freq[i]
		doc_num = word_doc_position[i]
		for k in range(0,len(term_freq)):
			doc_length = term_doc_rep[doc_num[k],:].sum(axis=None)
			score = rank_compute(word_doc_freq[i],term_freq[k],total_docs,avg_doc_length,doc_length)
			final_scores[doc_num[k]]+=score
	return final_scores	

def rank_compute(n,f,N,avdl,dl):    
    k1 = 1.5
    k2 = 100
    b = 0.5
    R = 0.0
    qf=1
    r=0

    K = k1 * ((1-b) + b * (float(dl)/float(avdl)) )
    first = math.log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)
    return first * second * third   

def cosine_similairty_result_print(query_response, tfidf_rep):
	cos_sim = cosine_similarity(query_response, tfidf_rep)
	cos_sim = max(cos_sim)
	list_index=sorted(range(len(cos_sim)), key=lambda i: cos_sim[i])[-10:]
	list_index.reverse()
	for i in list_index:
		print(HashIndex_DocName[i],cos_sim[i],i)

if __name__ == "__main__":
	#index, value = max(enumerate(cos_sim), key=operator.itemgetter(1))
	#print doc_list[index]

	filled_col_ind=[]
	
	#read_S08() #midsem corpus test
	#read_S09() #midsem corpus test
	#read_S10() #midsem corpus test
	
	file_read_from_corpus('corpus.txt')
	stemmer = Porter2Stemmer()
	Query_String = "Perform evaluation and model of computer system "
	q = Query_String.split(" ")
	QueryString = []
	for i in q:
		QueryString.append(stemmer.stem(i))
	QueryString = " ".join(QueryString) 
	#print QueryString

	term_doc_rep = tf_matrix(HashedDocuments_List) #term document matrix 	
	tfidf_matrix,tfidf_rep,query_response = tfidf(HashedDocuments_List,QueryString) #tfidf_matrix, query_vector
	print term_doc_rep.shape,tfidf_rep.shape,query_response.shape
	
	qm = scipy.sparse.coo_matrix(query_response)
	
	for i in qm.col:
		filled_col_ind.append(i)

	final_scores = bm25_function(filled_col_ind,tfidf_rep,term_doc_rep)
	sorted_Res = sorted(final_scores.items(),key = operator.itemgetter(1))
	print sorted_Res
		
	#bm25_result_print(filled_col_ind,tfidf_rep,term_doc_rep) #print bm25 result
	cosine_similairty_result_print(query_response, tfidf_rep) #cosine similarity results print

	

	


	





	
	
	
	

