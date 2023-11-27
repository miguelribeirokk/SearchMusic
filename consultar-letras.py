import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import re
from unidecode import unidecode
#%%
def preprocess(text):
    # Remover acentos
    text = unidecode(text)

    # Substituir caracteres não alfabéticos (incluindo vírgula) por espaços em branco
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Converter para minúsculas e dividir em palavras
    tokens = text.lower().split()

    return ' '.join(tokens)
#%%
def load_lyrics():
    raw_lyrics = joblib.load('lyrics_vector.joblib')
    return [preprocess(text) for text in raw_lyrics]


#%%
def create_tfidf_matrix(lyrics):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lyrics)
    return tfidf_matrix, vectorizer

#%%
def perform_query(query, vectorizer, tfidf_matrix, lyrics):
    preprocessed_query = preprocess(query)
    print(f"\nConsulta: {preprocessed_query}")
    query_vector = vectorizer.transform([preprocessed_query])

    # Calcular similaridade
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()

    # Adicionar um fator de prioridade para palavras em sequência
    sequence_priority = 100.0  # Ajuste conforme necessário
    for i in range(len(lyrics)):
        document = lyrics[i]
        occurrences = document.lower().count(preprocessed_query)
        cosine_similarities[i] += sequence_priority * occurrences

    document_scores = list(enumerate(cosine_similarities))
    sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)

    # Imprimir resultados
    print("\nResultados:")
    for index, score in sorted_documents[:5]:
        print(
            f"Arquivo: {os.listdir('C:/Users/miguel/Documents/UFV/6 PERIODO/GRADI/musics-txt')[index]}, Pontuação: {score}")

#%%

# Carregar vetor de letras
print("Carregando letras...")
lyrics = load_lyrics()

# Criar matriz TF-IDF
print("Criando matriz TF-IDF...")
tfidf_matrix, vectorizer = create_tfidf_matrix(lyrics)
print(f"Dimensões da matriz TF-IDF: {tfidf_matrix.shape}")
#%%
def main():
    # Menu de inserção de consulta
    while True:
        query = input("\nInsira sua consulta (ou 'exit' para sair): ")
        if query.lower() == 'exit':
            break
        perform_query(query, vectorizer, tfidf_matrix, lyrics)


if __name__ == "__main__":
    main()