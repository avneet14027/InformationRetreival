from __future__ import division
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import re

tokenize = lambda doc: doc.lower().split(" ")
qa_pairs = []
qa_pairs_doc = []

#file read, sore docs in list
def file_read(filename):
	doc_list = []
	with open(filename) as f:
		f.readline()
		count_line=2
		for line in f:
			line = unicode(line,errors='ignore')
			line1 = line.lower().rstrip()
			#print line
			new_array = []
			array = line1.split('\t')
			#print array
			answer = array[2]
			if(answer!='no' and answer!='yes' and answer!='null' and answer!='yeah' and len(answer.split())>1):
				new_array.append(array[1])#.translate(None, string.punctuation))
				new_array.append(answer)
				new_array.append(array[5])#.translate(None, '\t\n'))
				doc_list.append(new_array)
				qa_pairs.append(answer)
				qa_pairs_doc.append(filename+"/"+str(count_line))
				#print(new_array)
			else:
				pass
			count_line+=1
			
		
	f.close()
	return doc_list

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

if __name__ == "__main__":
	file_read('question_answer_pairs1.txt')
	file_read('question_answer_pairs2.txt')
	file_read('question_answer_pairs3.txt')
	#print qa_pairs
	#print qa_pairs_doc


	tf_idf_matrix = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
	tf_idf_rep = tf_idf_matrix.fit_transform(qa_pairs)
	Query_String = "Was Alessandro Volta taught in public schools?"
	Query_resp = tf_idf_matrix.transform([Query_String])
	cos_sim = cosine_similarity(Query_resp, tf_idf_rep)
	cos_sim = cos_sim[0]
	a=sorted(range(len(cos_sim)), key=lambda i: cos_sim[i])[-8:]
	a.reverse()
	for i in a:
		print("Document and line number: ",qa_pairs_doc[i],"Cosine similarity: ",cos_sim[i])



