from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data1 = word_tokenize("trời hôm nay thế nào".lower())
test_data2 = word_tokenize("thời tiết hôm nay thế nào".lower())

v1 = model.infer_vector(test_data1)
print("V1_infer", v1)

v2 = model.infer_vector(test_data2)
print("V2_infer", v2)

#dis = model.docvecs.distance([v1] , [v2])