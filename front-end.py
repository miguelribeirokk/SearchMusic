import os
import re
import tkinter as tk
from tkinter import Scrollbar, Text, END, filedialog
from tkinter import messagebox


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode


# Função de pré-processamento
def preprocess(text):
    # Remover acentos
    text = unidecode(text)

    # Substituir caracteres não alfabéticos (incluindo vírgula) por espaços em branco
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Converter para minúsculas e dividir em palavras
    tokens = text.lower().split()

    return ' '.join(tokens)

def vector_music_name():
    music_name = []
    files = os.listdir('./arquivos')
    for file in files:
        if file.endswith(".txt"):
            music_name.append(file)
    return music_name



# Função para carregar letras de arquivos
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
                #print(f"Arquivo {file} carregado")

    return lyrics


# Função para criar matriz TF-IDF
def create_tfidf_matrix(lyrics):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lyrics)
    return tfidf_matrix, vectorizer


# Função para realizar consulta e exibir resultados
def perform_query():
    query = entry_query.get()
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

    # Limpar a área de texto
    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', END)

    # Adicionar resultados à área de texto
    for index, score in sorted_documents[:5]:
        result_text.insert(tk.END, f"Música: ", "bold")
        result_text.insert(tk.END, f"{music_names[index]}", "normal")
        result_text.insert(tk.END, f", Pontuação: ", "bold")
        result_text.insert(tk.END, f"{score}\n\n", "normal")

    # Desativar edição na área de texto
    result_text.config(state=tk.DISABLED)


# Função para adicionar arquivo à pasta e recalcular TF-IDF
def add_file_and_recalculate():
    global tfidf_matrix, vectorizer  # Adicione esta linha
    file_path = filedialog.askopenfilename(title="Selecione um arquivo TXT", filetypes=[("Arquivos de Texto", "*.txt")])

    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            print(text)
            preprocessed_text = preprocess(text)
            lyrics.append(preprocessed_text)
            print(f"Arquivo {os.path.basename(file_path)} adicionado")

            # Salvar o arquivo na pasta /arquivos
            dest_path = os.path.join('./arquivos', os.path.basename(file_path))
            with open(dest_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write(text)
                print(f"Arquivo {os.path.basename(file_path)} salvo em /arquivos")

        music_names.append(os.path.basename(file_path))
        print(music_names)

        # Recalcular TF-IDF
        tfidf_matrix, vectorizer = create_tfidf_matrix(lyrics)
        print(f"Dimensões da matriz TF-IDF: {tfidf_matrix.shape}")

        messagebox.showinfo("Sucesso", f"Arquivo {os.path.basename(file_path)} adicionado com sucesso!")



print("Carregando letras...")
lyrics = load_lyrics('./arquivos')
music_names = vector_music_name()

# Criar matriz TF-IDF
print("Criando matriz TF-IDF...")
tfidf_matrix, vectorizer = create_tfidf_matrix(lyrics)
print(f"Dimensões da matriz TF-IDF: {tfidf_matrix.shape}")

# Configurar a interface Tkinter
root = tk.Tk()
root.title("Consulta de Letras")

# Widgets da interface
label_query = tk.Label(root, text="Insira sua consulta:")
entry_query = tk.Entry(root, width=50)
button_search = tk.Button(root, text="Buscar", command=perform_query)
button_add_file = tk.Button(root, text="Adicionar música", command=add_file_and_recalculate)
result_text = Text(root, wrap='word', width=80, height=20, state=tk.DISABLED)
scrollbar = Scrollbar(root, command=result_text.yview)

# Layout dos widgets
label_query.pack(pady=10)
entry_query.pack(pady=5)
button_search.pack(pady=5)
button_add_file.pack(pady=5)
result_text.pack(pady=10)
scrollbar.pack(side='right', fill='y')

# Configurar a tag de formatação
result_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
result_text.tag_configure("normal", font=("Helvetica", 10, "normal"))



# Configurar scrollbar para rolar o Text widget
result_text.config(yscrollcommand=scrollbar.set)

# Iniciar a interface Tkinter
root.mainloop()
