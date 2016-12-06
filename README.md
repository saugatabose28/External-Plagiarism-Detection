# External-Plagiarism-Detection
./plagiarismDetector.py

Used framework: NLTK,sklearn,numpy,weka
Steps A: A Train model has been created based on 95 suspicous documents and 5 orginal documents
Steps:
1. Preprocessing: 
a)Lowercasing
b)Uppercasing
c)Erasing extra space,newlines and ':'s.
d)Stemming using PorterStemmer()
e)Lemmatzing using WordNetLemmatizer()
f)Removing only stop words
g)Removing only punctations
h)Removing stops words as well as punctuations
2.Comparing Documents
     Document Summery:Using n gram feature, vocabulary frequencies have been calculated based on 4 preprocessing features(with stopword-pnctuation,without stopword-punctuation,only stop swords and only pnctuations)
     Relative Frequency: n gram words have been compared and number of common words have been calculated as well as the frequency
     Jaccard Similarty:Jaccardian similarity between docments have been calculated
     
We save these smilarty scores in a csv file.

Steps B: We use this train model data to calculate the similarty score.
Here we use J48 classifier
     
