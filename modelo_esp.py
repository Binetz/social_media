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
from scipy import sparse
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


def get_ypos(xtrain, xvect=None):
  return(xvect if xvect is not None else np.zeros(rows))


def iter_fit(x_train, y_train, y_pos, prob_umbral, x_iter, n_estimators=10):
  x_modelo = RandomForestClassifier(
  n_estimators      = n_estimators, # cantidad de arboles a crear
  min_samples_split = 4,   # cantidad minima de observaciones para dividir un nodo
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
    y_pos2 = get_ypos(pred_proba, y_pos)
    pred_proba.loc[y_pos2 == 1, ('ytrain')] = 0
    y_train2 = pred_proba['ytrain']
    modelo.fit(X = x_train, y = y_train2)
  return modelo




def df_to_numeric(df):
  for colname in df.columns:
    if df[colname].dtype == 'O':
      df[colname] = df[colname].str.replace(",", ".")
      df[colname] = df[colname].str.replace(" ", "")
  return df.apply(pd.to_numeric)

def df_to_csr(df):
  #convierte pandas dataframe en matriz dispersa csr
  df = df_to_numeric(df)
  return sparse.csr_matrix(df.values)

def concat_csr_df(x_csr, x_df):
  x_df = df_to_csr(x_df.drop(no_feature, axis=1))
  return hstack([x_csr, x_df])



# EJECUCION
# ----------------------------------------------------------------------------
# 1. CARGAR DATOS DE DROPBOX
data_path = '..'
xdata = pd.read_csv(data_path + 'CSV_SENTIMENT/coment_feature.csv').iloc[0:, ]
xdata['coment_orig'] = clean_corpus(xdata['coment_orig'])
ydata = xdata['clase']
coment_pos_manual = xdata['coment_pos_manual']
xdata = xdata.drop('clase', axis= 1)
no_feature = ['%Comentario', 'coment_orig', 'coment_pos_manual']

# 2. TRAIN TEST
#xtrain, xtest, ytrain, ytest = tts(xdata, ydata, train_size=0.70)
xtrain, xtest, ytrain, ytest = xdata, xdata, ydata, ydata


# 3. TOKENIZACION + VECTORIZACION + FEATURES
xcorpus_train, xcorpus_test, vectorizer = train_test_vector(xtrain=xtrain['coment_orig'], xtest=xtest['coment_orig'])
xcorpus_train = concat_csr_df(xcorpus_train, xtrain)
xcorpus_test = concat_csr_df(xcorpus_test, xtest)

# 4. MODELO
xmodelo = iter_fit(xcorpus_train, ytrain, coment_pos_manual, prob_umbral=0.15, x_iter=2, n_estimators=102)

# 5. PREDICT + METRICAS
prediccion = xmodelo.predict(xcorpus_test)
pred_proba = xmodelo.predict_proba(xcorpus_test)
pred_proba = pd.DataFrame(pred_proba, columns=['pos', 'neg'])
pred_proba['%Comentario'] = xtest['%Comentario']
pred_proba.to_csv(data_path + 'CSV_SENTIMENT/pred_proba.csv', index=False)

print(pd.crosstab(ytest, prediccion, rownames=['REAL'], colnames=['PREDICCION']))
print(classification_report(ytest, prediccion))

