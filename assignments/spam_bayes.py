import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dataFrame = pd.read_table('datasets/SMSSpamCollection', sep='\t', names=["labels","data"])

documents = ["Hi, how are you","call me if free ?", "Hello World"]

train_input, test_input, train_labels, test_labels = train_test_split(dataFrame["data"], dataFrame["labels"], random_state=1)

count_vector = CountVectorizer()
naive_bayes = MultinomialNB()

training_data = count_vector.fit_transform(train_input)
testing_data = count_vector.transform(test_input)

naive_bayes.fit(training_data,train_labels)

predictions = naive_bayes.predict(testing_data)

print "Accuracy: {}".format(accuracy_score(test_labels, predictions))
print "Precision: {}".format(precision_score(test_labels, predictions, pos_label="spam"))
print "Recall: {}".format(recall_score(test_labels, predictions, pos_label="spam"))
print "F1 Score: {}".format(f1_score(test_labels, predictions, pos_label="spam"))
