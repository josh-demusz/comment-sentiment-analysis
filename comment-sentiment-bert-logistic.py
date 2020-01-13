import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from bert_serving.client import BertClient
from keras.models import Sequential
from keras import layers

def print_metrics(y_test, y_predict, model_name):
    print('{} metrics'.format(model_name))
    print(classification_report(y_test, y_predict))


filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb':   'data/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])
print(df.size)

sentences = df['sentence'].values
labels = df['label'].values

max_sentence_length = df['sentence'].map(len).max()
print('Max sentence length:', max_sentence_length)

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.3, random_state=1000)

bc = BertClient(show_server_config=True)

X_train = bc.encode(sentences_train.tolist()) #for sentence in sentences_train]
X_test = bc.encode(sentences_test.tolist()) #for sentence in sentences_test]

print('Finished encoding')

# Train & test Logistic Regression model

classifier = LogisticRegression(solver='lbfgs', multi_class='auto', verbose=1)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
y_predict = classifier.predict(X_test)

print_metrics(y_test, y_predict, 'Logistic Regression')

score_train = classifier.score(X_train, y_train)
score_test = classifier.score(X_test, y_test)

print("Training Accuracy: {:.4f}".format(score_train))
print("Testing Accuracy: {:.4f}".format(score_test))

# Train & test Deep Neural Network
num_features = X_train.shape[1]

classifier = Sequential()
classifier.add(layers.Dense(10, input_dim=num_features, activation='relu'))
classifier.add(layers.Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
classifier.summary()

history = classifier.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

y_predict = classifier.predict_classes(X_test)

y_predict = [prediction[0] for prediction in y_predict]

print_metrics(y_test, y_predict, 'Deep Neural Network')

loss, accuracy = classifier.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = classifier.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))