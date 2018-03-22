from __future__ import division
import string
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#file read, store docs in list

tokenize = lambda doc: doc.lower().split(" ")

LabelDocs = []
def file_read(filename,doc_list):
	document = []
	with open(filename) as f:
    		document = f.read()
		document = unicode(document,errors='ignore')
	MainDocuments_List.append(document)

def read_S08():
	for i in range(1,5):
		for j in range(1,11):
			string = "S08/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			LabelDocs.append(string)
			file_read(string,MainDocuments_List)

def read_S09():
	for i in range(1,6):
		for j in range(1,11):
			string = "S09/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			LabelDocs.append(string)
			file_read(string,MainDocuments_List)


def read_S10():
	for i in range(1,7):
		for j in range(1,11):
			string = "S10/data/set" + str(i) + "/a" + str(j) + ".txt.clean"
			LabelDocs.append(string)
			file_read(string,MainDocuments_List)

#######	
def construct_index(Words):
	set_words = []
	for word in Words:
		if word not in set_words:
			set_words.append(word)

	#print set_words, len(set_words)
	
	dict_words = {}
	for word in set_words:
		indices = [n for (n, e) in enumerate(Words) if e == word] #get word position
		dict_words[word] = indices
	#print dict_words
	return dict_words


def construct_main_index(dict_words,main_index,ctr):
	for word in dict_words:
		if word not in main_index:
			main_index[word] = {}
			
		main_index[word][ctr]=dict_words[word]
	#print main_index
	main_index = collections.OrderedDict(sorted(main_index.items()))
#######

if __name__ == "__main__":
	MainDocuments_List=[]
	read_S08()
	read_S09()
	read_S10()

	tfidf_matrix= TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
	tfidf_rep = tfidf_matrix.fit_transform(MainDocuments_List)

	QueryString = "When did the Gettysburg address argue that America was born?"
	query_reponse = tfidf_matrix.transform([QueryString])
	cos_sim = cosine_similarity(query_reponse, tfidf_rep)
	cos_sim = max(cos_sim)
	a=sorted(range(len(cos_sim)), key=lambda i: cos_sim[i])[-8:]

	a.reverse()
	for i in a:
		print("Document name and cosine similarity ",LabelDocs[i],cos_sim[i])
	
	#index, value = max(enumerate(cos_sim), key=operator.itemgetter(1))
	#print doc_list[index]
	
	
	
	

