import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode


# %%
def preprocess(text):
    text = unidecode(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    return ' '.join(tokens)


# %%
def load_lyrics(directory_path):
    print(f"Carregando letras de {directory_path}...")
    lyrics = []
    files = os.listdir(directory_path)

    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                text = f.read()
                preprocessed_text = preprocess(text)
                lyrics.append(preprocessed_text)
                print(f"Arquivo {file} carregado")
    return lyrics


# %%
def create_tfidf_matrix(lyrics):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lyrics)
    return tfidf_matrix, vectorizer


# %%
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
            f"Arquivo: {os.listdir('./arquivos')[index]}, Pontuação: {score}")


# %%

print("Carregando letras...")
lyrics = load_lyrics('./arquivos')

print("Criando matriz TF-IDF...")
tfidf_matrix, vectorizer = create_tfidf_matrix(lyrics)
print("Portanto, para o primeiro documento (linha 0) e a palavra na coluna 5629, o valor TF-IDF é 0.028483012796560816")

print(f"Dimensões da matriz TF-IDF: {tfidf_matrix.shape}")
print(tfidf_matrix)

feature_names = vectorizer.get_feature_names_out()
word_at_index_5629 = feature_names[5629]

print(f"A palavra associada ao índice 5629 é: {word_at_index_5629}")


# %%
def main():
    # Menu de inserção de consulta
    while True:
        query = input("\nInsira sua consulta (ou 'exit' para sair): ")
        if query.lower() == 'exit':
            break
        perform_query(query, vectorizer, tfidf_matrix, lyrics)


if __name__ == "__main__":
    main()
