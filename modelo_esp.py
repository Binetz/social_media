import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
import nltk 
from sklearn.cross_validation import train_test_split as tts
import re
from scipy.sparse import coo_matrix, hstack
#from sklearn.neural_network import MLPClassifier


# FUNCIONES
# ----------------------------------------------------------------------------
def clean_corpus(corpus):
  xcorpus = corpus.get_values()
  for i in range(len(corpus)):
    xcorpus[i] = str(xcorpus[i])
    xcorpus[i] = re.sub("[^a-zA-Z]", " ", corpus[i].lower())
    xcorpus[i] = ' '.join(xcorpus[i].split())
  return xcorpus

def tokenize(text):
    tokens = nltk.word_tokenize(text,  language='spanish')
    stems = []
    for item in tokens:
        stems.append(nltk.PorterStemmer().stem(item))
    return stems

def train_test_vector(xtrain, xtest):
   vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df = 0.50, use_idf=True,min_df=6)
   #vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df = 0.70, use_idf=True,min_df=3)
   #vectorizer    = CountVectorizer(min_df=1,binary=True) # metrica binaria
   vector_train  = vectorizer.fit_transform(xtrain)
   vector_test   = vectorizer.transform(xtest)
   return vector_train, vector_test, vectorizer


def iter_fit(x_train, y_train, prob_umbral, x_iter, n_estimators=10):
  x_modelo = RandomForestClassifier(
  n_estimators      = n_estimators, # cantidad de arboles a crear
  min_samples_split = 2,   # cantidad minima de observaciones para dividir un nodo
  min_samples_leaf  = 1,   # observaciones minimas que puede tener una hoja del arbol
  n_jobs            = -1    # tareas en paralelo. para todos los cores disponibles usar -1
  )
  modelo = x_modelo
  modelo.fit(X = x_train, y = y_train)
  for i in range(x_iter):
    pred_proba = modelo.predict_proba(x_train)
    pred_proba = pd.DataFrame(pred_proba, columns=['pos', 'neg'])
    pred_proba['ytrain'] = y_train.values
    pred_proba.loc[pred_proba['neg'] >= prob_umbral, ('ytrain')] = 1
    y_train2 = pred_proba['ytrain']
    modelo.fit(X = x_train, y = y_train2)
  return modelo





# EJECUCION
# ----------------------------------------------------------------------------
# PATH
data_path = '..'
# 1. CARGAR DATOS DE DROPBOX
xdata = pd.read_csv(data_path + 'CSV_SENTIMENT/coment_feature.csv').iloc[:, ]
xdata['coment_orig'] = clean_corpus(xdata['coment_orig'])
ydata = xdata['clase']
xdata = xdata.drop('clase', axis= 1)
no_feature = ['%Comentario', '%Publicacion', 'Fecha', 'Hora', 'coment_orig']

# 3. TRAIN TEST
#xtrain, xtest, ytrain, ytest = tts(xdata, ydata, train_size=0.70)
xtrain, xtest, ytrain, ytest = xdata, xdata, ydata, ydata 



# 4. TOKENIZACION + VECTORIZACION
xcorpus_train = xtrain['coment_orig']
xcorpus_test = xtest['coment_orig']
xcorpus_train, xcorpus_test, vectorizer = train_test_vector(xtrain=xcorpus_train, xtest=xcorpus_test)

"""
ERROR MEMORIA EN sparse to dense
xcorpus_train = xcorpus_train.A
xcorpus_train = xcorpus_test.A
xtrain = xtrain.drop(no_feature, axis=1)
xtrain = np.array(xtrain)
xtest = xtest.drop(no_feature, axis=1)
xcorpus_train = np.c_[xcorpus_train, xtrain]
xcorpus_test = np.c_[xcorpus_test, xtest]
"""

# 5. MODELO
"""
modelo = svm.SVC(kernel='linear') 
modelo.fit(X=xtrain, y=ytrain)
"""


"""
xmodelo = RandomForestClassifier(
 n_estimators      = 66, # cantidad de arboles a crear
 min_samples_split = 2,   # cantidad minima de observaciones para dividir un nodo
 min_samples_leaf  = 1,   # observaciones minimas que puede tener una hoja del arbol
 n_jobs            = -1    # tareas en paralelo. para todos los cores disponibles usar -1
 )
xmodelo.fit(X = xcorpus_train, y = ytrain)
"""

xmodelo = iter_fit(xcorpus_train, ytrain, prob_umbral=0.50, x_iter=2, n_estimators=333)

# 6. PREDICT + METRICAS
prediccion = xmodelo.predict(xcorpus_test)
pred_proba = xmodelo.predict_proba(xcorpus_test)
pred_proba = pd.DataFrame(pred_proba, columns=['pos', 'neg'])
pred_proba['%Comentario'] = xtest['%Comentario']
pred_proba.to_csv(data_path + 'CSV_SENTIMENT/pred_proba.csv', index=False)

print(pd.crosstab(ytest, prediccion, rownames=['REAL'], colnames=['PREDICCION']))
print(classification_report(ytest, prediccion))

