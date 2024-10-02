import pandas as pd
import numpy as np
import string
import pprint
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def converterTudoEmMinuscula (doc):
    lower_case_documents = []
    for linha in doc:
        lower_case_documents.append(linha.lower())
    return lower_case_documents

def retirarTodosOsSinaisDePontuacao(doc):
    documento_sem_sinais = []
    for linha in doc:
        linha_sem_sinais = []
        linha_sem_sinais = ''.join(caracter for caracter in linha if caracter not in string.punctuation)
        documento_sem_sinais.append(linha_sem_sinais)
    return documento_sem_sinais

def separarTodasAsPalavras(doc):
    palavrasSeparadas = []
    for linha in doc:
        palavrasSeparadas.append(linha.split())
    return palavrasSeparadas

def preprocessarDocumentos(doc):
    return separarTodasAsPalavras(retirarTodosOsSinaisDePontuacao(converterTudoEmMinuscula(doc)))

def gerarListaDeContadoresPorDocumento(doc):
    """Gera uma list cujos elementos são contadores de palavras para cada documento."""
    contadoresDePalavrasPorLinha = []
    for linha in doc:
        contagem = Counter(linha)
        contadoresDePalavrasPorLinha.append(contagem)
    return contadoresDePalavrasPorLinha

def gerarListaGeralDePalavras(contadores):
    """Gera uma list contendo todas as palavras, sem repetição, da união de todos os documentos
        a partir dos contadores gerados para cada documento"""
    conjunto = set()
    for contador in contadores:
        for chave in contador.keys():
            conjunto.add(chave)
    return list(conjunto)

def gerarDataFrameDeFrequencias(colunas, linhas):
    df = pd.DataFrame(columns=colunas)
    for linha in linhas:
        df=pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
    df.fillna(0, inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def substituirHamSpamPor01(df):
    mapeamento = {'ham':0, 'spam':1}
    df['label'] = df['label'].map(mapeamento)
    return df

def exibirInformacoesDoVocabulario(vocabulario):
    print('Quantidade de palavras no vocabulário: {}'.format(len(vocabulario.vocabulary_)))
    print('Palavras no vocabulário seguidas de seus índices:\n{}'.format(vocabulario.vocabulary_))

def gerarColunasOrdenadas(dicionario):
    """Ordena o dicionario retornado por CounterVectorizer.vocabulary_ de acordo com values do dicionario, que
    são os índices de cada palavra no vocabulário, para assim haver correspondência entre as colunas da matriz 
    retornada  pelo método toarray() de um objeto retornado por CounterVectorizer.transform() com as palavras
    do vocabulário. Após a ordenação, gera uma list contendo somente as keys já ordenadas. 
    Essa lista será utilizada para nomear as colunas de um DataFrame."""
    dicionario_ordenado =  dict(sorted(dicionario.items(), key = lambda item: item[1]))
    listaDeColunasOrdenadas = list(dicionario_ordenado.keys())
    return listaDeColunasOrdenadas

def gerarDataFrameVocabulario(dados, colunas):
    df = pd.DataFrame(dados, columns=colunas)
    return df

def calcularProbabilidadeDeTerTestePositivo():
    p_diabetes = 0.01
    p_no_diabetes = 0.99
    p_pos_diabetes = 0.9
    p_neg_no_diabetes = 0.9
    p_pos = p_diabetes * p_pos_diabetes + (1 - p_diabetes)*(1-p_neg_no_diabetes)
    return p_pos

def calcularProbabilidadeDeNaoTerDiabetesDadoResultadoPositivo():
    return 0.99*0.1/calcularProbabilidadeDeTerTestePositivo()

# 1) ==========ÁREA QUE TRATA A BASE DE DADOS SMSSPAMCOLLECTION=====================================
def executarOQueFoiPedidoSobreSmsSpamCollection():
    FILE = 'SMSSpamCollection'
    df = pd.read_table(FILE, header=None, names=['label', 'sms_message'])
    df = substituirHamSpamPor01(df)

    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)
    print('Number of rows in the total set: {}'.format(df.shape[0]))
    print('Number of rows in the training set: {}'.format(X_train.shape[0]))
    print('Number of rows in the test set: {}'.format(X_test.shape[0]))

    count_vector = CountVectorizer()
    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)

    naive_bayes =  MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    predictions = naive_bayes.predict(X_test)
 

#TODO
executarOQueFoiPedidoSobreSmsSpamCollection()

# 2) ==========ÁREA QUE TRATA A LIST DOCUMENTS======================================================
def executarOQueFoiPedidoSobreDocumentsFromScratch():
    documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call you tomorrow?']

    preprocessed_documents = preprocessarDocumentos(documents)
    contadores = gerarListaDeContadoresPorDocumento(preprocessed_documents)
    colunas = gerarListaGeralDePalavras(contadores)
    df2 = gerarDataFrameDeFrequencias(colunas,contadores)
    pprint.pprint(df2)

#TODO
#executarOQueFoiPedidoSobreDocumentsFromScratch()

# 2.2) ..........Usando o package sklearn
def executarOQueFoiPedidoSobreDocumentsComSklearn():
    documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call you tomorrow?']

    count_vector3 = CountVectorizer()
    count_vector3.fit(documents)
    exibirInformacoesDoVocabulario(count_vector3)
    vocabulario = count_vector3.transform(documents)
    df3 = gerarDataFrameVocabulario(vocabulario.toarray(), gerarColunasOrdenadas(count_vector3.vocabulary_))
    print(df3)

# TODO
#executarOQueFoiPedidoSobreDocumentsComSklearn()

# 3) ==========ÁREA QUE TRATA DO TEOREMA DE BAYES=====================
def executarOQueFoiPedidoSobreTeoremaDeBayes():
    print('The probability of getting a positive test result P(pos) is: {}'.format(calcularProbabilidadeDeTerTestePositivo()))
    print('Probability of an individual not having diabetes, given that individual got a positive test result is:{}'.format(
        calcularProbabilidadeDeNaoTerDiabetesDadoResultadoPositivo()))

    """
    REFLEXÃO: As chances de uma pessoa não ter diabetes dado que ela recebeu um teste positivo para diabetes é em torno de
    91.67% caso a pessoa tenho sido escolhida aleatoriamente para fazer o teste. Em uma situação real, se uma pessoa vai
    fazer o teste, em muitos casos, é porque ela suspeita de algo ou por indicação de um clínico geral que percebeu 
    determinados sintomas e suspeita de que o paciente tenha diabetes, então, acredito eu, as chances de a pessoa 
    não ter diabetes são menores caso ela tenho recebido um teste positivo. Está certo este raciocínio?
    """

    p_j = 0.5
    p_j_f=0.1
    p_j_i=0.1
    p_j_text = p_j*p_j_f * p_j_i
    print(p_j_text)

    p_g = 0.5
    p_g_f = 0.7
    p_g_i = 0.2
    p_g_text = p_g*p_g_f*p_g_i
    print(p_g_text)

    p_f_i = p_j_text + p_g_text
    print('Probability of words freedom and immigration being said are: {}'.format(p_f_i))

    """Aqui eu não entendi direito: P(J|F,I) é a probabilidade de Jill fazer um discurso com  as palavras Freedom e Immigration     ou é a probabilidade de Jill ter feito o discurso dado que as palavras Freedom e Immigration foram pronunciadas? 
    Esses dois modos de interpretar são idênticos? A mesma dúvida ficou em relação a Gary."""

    p_j_fi = (p_j * p_j_f * p_j_i)/p_f_i
    print('The probability of Jill Stein saying the words Freedom and Immigration: {}.'.format(p_j_fi))

    p_g_fi = (p_g * p_g_f * p_g_i)/p_f_i
    print('The probability of Gary Johnson saying the words Freedom and Immigration: {}.'.format(p_g_fi))
#TODO
#executarOQueFoiPedidoSobreTeoremaDeBayes()


