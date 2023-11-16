import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import re


# Função de pré-processamento
def preprocess(text):
    # Substituir caracteres não alfabéticos por espaços em branco
    text = re.sub(r'[^a-zA-Zá-úÁ-Ú]', ' ', text)

    # Converter para minúsculas e dividir em palavras
    tokens = text.lower().split()

    return ' '.join(tokens)


# Função para carregar as letras
def load_lyrics():
    return joblib.load('lyrics_vector.joblib')


# Função para criar a matriz TF-IDF
def create_tfidf_matrix(lyrics):
    preprocessed_lyrics = [preprocess(text) for text in lyrics]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_lyrics)
    return tfidf_matrix, vectorizer


# Função para realizar uma consulta
def perform_query(query, vectorizer, tfidf_matrix, lyrics):
    preprocessed_query = preprocess(query)
    print(f"\nConsulta: {preprocessed_query}")
    query_vector = vectorizer.transform([preprocessed_query])

    # Calcular similaridade
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()

    # Contar o número de ocorrências da sequência na consulta
    sequence_count = preprocessed_query.count(' ')

    if sequence_count == 0:
        sequence_count = 1

    # Adicionar um fator de prioridade para palavras em sequência
    sequence_priority = 100.0  # Ajuste conforme necessário
    for i in range(len(lyrics)):
        document = lyrics[i]
        occurrences = document.lower().count(preprocessed_query)
        cosine_similarities[i] += sequence_priority * occurrences / sequence_count

    document_scores = list(enumerate(cosine_similarities))
    sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)

    # Imprimir resultados
    print("\nResultados:")
    for index, score in sorted_documents[:5]:
        print(
            f"Arquivo: {os.listdir('C:/Users/miguel/Documents/UFV/6 PERIODO/GRADI/musics-txt')[index]}, Pontuação: {score}")



# Função principal
def main():
    # Carregar vetor de letras
    print("Carregando letras...")
    lyrics = load_lyrics()

    # Criar matriz TF-IDF
    print("Criando matriz TF-IDF...")
    tfidf_matrix, vectorizer = create_tfidf_matrix(lyrics)
    print(f"Dimensões da matriz TF-IDF: {tfidf_matrix.shape}")

    # Menu de inserção de consulta
    while True:
        query = input("\nInsira sua consulta (ou 'exit' para sair): ")
        if query.lower() == 'exit':
            break
        perform_query(query, vectorizer, tfidf_matrix, lyrics)


if __name__ == "__main__":
    main()

#%%
