
# Operaciones CRUD con ChromaDB y Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Snickmax/CrudEmmbedding/blob/main/CrudEmbedding.ipynb)

Este proyecto muestra cómo configurar una base de datos vectorial utilizando ChromaDB con embeddings generados por modelos de lenguaje. A continuación, se detalla cómo instalar, configurar y realizar operaciones CRUD, junto con ejemplos prácticos detallados.

---

## Instalación

### Requisitos
- Python 3.8 o superior
- pip para manejar las dependencias

### Instalación de dependencias
Ejecuta el siguiente comando para instalar las bibliotecas necesarias:
```bash
pip install chromadb langchain sentence-transformers numpy
```

---

## Configuración de la base de datos vectorial

La configuración de la base de datos incluye los siguientes pasos:

1. **Modelo de embeddings**: Utilizamos `sentence-transformers/all-MiniLM-L6-v2` para convertir textos en vectores. Este modelo genera representaciones numéricas de textos que capturan su significado semántico.
2. **Creación del directorio de almacenamiento**: La base de datos se almacena localmente en el directorio `chroma_db`.
3. **Inicialización de la base de datos**: Configuramos ChromaDB con el modelo de embeddings.

Código para la configuración inicial:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Configurar el modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear directorio y configurar la base de datos
directorio = "chroma_db"
os.makedirs(directorio, exist_ok=True)
base_datos = Chroma(persist_directory=directorio, embedding_function=embedding_model)
print("Base de datos configurada.")
```

---

## Operaciones CRUD

CRUD se refiere a las operaciones de Crear, Leer, Actualizar y Eliminar documentos en la base de datos. A continuación, se describen en detalle:

### 1. **Crear: Agregar documentos**

Para agregar documentos, se vectoriza el texto proporcionado y se almacena junto con el contenido original. Esto permite realizar búsquedas basadas en similitud más adelante.

Código para agregar un documento:
```python
from langchain.schema import Document

class DocumentManager:
    def __init__(self, directorio="chroma_db"):
        self.directorio = directorio
        self.base_datos = self.configurar_base_datos(directorio)

    def configurar_base_datos(self, directorio):
        os.makedirs(directorio, exist_ok=True)
        base_datos = Chroma(persist_directory=directorio, embedding_function=embedding_model)
        print(f"Base de datos configurada en: {directorio}")
        return base_datos

    def agregar_documento(self, texto):
        # Generar un ID único para el documento
        id_documento = str(hash(texto))  # Usamos el hash del texto como ID
        documento = Document(page_content=texto, metadata={"id": id_documento})
        self.base_datos.add_documents([documento])
        print(f"Documento agregado: '{texto}' con ID {id_documento}")
        return id_documento  # Devolvemos el ID para futuras referencias

# Ejemplo de uso
doc_manager = DocumentManager(directorio="chroma_db")
doc_manager.agregar_documento("Este es un texto de ejemplo.")

```

### 2. **Leer: Consultar documentos**

La lectura se realiza mediante consultas por similitud. Dado un texto, la base de datos devuelve los documentos más cercanos en términos semánticos.

Código para consultar documentos:
```python
def actualizar_documento(texto_viejo, texto_nuevo):
    resultados = self.base_datos.similarity_search(texto_viejo, k=1)  # Buscamos el documento más similar
    if resultados:
        documento_antiguo = resultados[0]
        id_documento = documento_antiguo.metadata.get("id")
        if id_documento:
            self.base_datos.delete([id_documento])  # Eliminamos el documento viejo
            doc_manager.agregar_documento(texto_nuevo)
            print(f"Documento actualizado: '{texto_viejo}' a '{texto_nuevo}'")
        else:
            print(f"Error: El documento con el texto '{texto_viejo}' no tiene un ID válido.")
    else:
        print(f"No se encontró el documento con el texto: '{texto_viejo}'")

# Ejemplo de uso
actualizar_documento("Este es un texto de ejemplo.", "Texto de ejemplo actualizado.")


```

### 3. **Actualizar: Modificar documentos**

La actualización de un documento implica eliminar el contenido existente y agregar el nuevo contenido con los cambios deseados.

Código para actualizar un documento:
```python
def actualizar_documento(texto_viejo, texto_nuevo):
    resultados = self.base_datos.similarity_search(texto_viejo, k=1)  # Buscamos el documento más similar
    if resultados:
        documento_antiguo = resultados[0]
        id_documento = documento_antiguo.metadata.get("id")
        if id_documento:
            self.base_datos.delete([id_documento])  # Eliminamos el documento viejo
            doc_manager.agregar_documento(texto_nuevo)
            print(f"Documento actualizado: '{texto_viejo}' a '{texto_nuevo}'")
        else:
            print(f"Error: El documento con el texto '{texto_viejo}' no tiene un ID válido.")
    else:
        print(f"No se encontró el documento con el texto: '{texto_viejo}'")

# Ejemplo de uso
actualizar_documento("Este es un texto de ejemplo.", "Texto de ejemplo actualizado.")

```

### 4. **Eliminar: Borrar documentos**

Para eliminar un documento, simplemente se proporciona el texto a borrar.

Código para eliminar un documento:
```python
def eliminar_documento(texto):
    resultados = self.base_datos.similarity_search(texto, k=1)  # Buscamos el documento más similar
    if resultados:
        documento = resultados[0]
        id_documento = documento.metadata.get("id")
        if id_documento:
            self.base_datos.delete([id_documento])
            print(f"Documento eliminado: '{texto}'")
        else:
            print(f"Error: El documento con el texto '{texto}' no tiene un ID válido.")
    else:
        print(f"No se encontró el documento con el texto: '{texto}'")

# Ejemplo de uso
eliminar_documento("Texto de ejemplo actualizado.")
```

---

## Funcionamiento de los Embeddings

### ¿Qué son los embeddings?
Los embeddings son representaciones vectoriales de textos que capturan su significado semántico. Utilizando modelos preentrenados como `sentence-transformers`, podemos convertir cualquier texto en un vector numérico de alta dimensión.

### Proceso de trabajo:
1. **Entrada**: Se proporciona un texto como entrada.
2. **Vectorización**: El texto pasa por un modelo de embeddings que genera su representación vectorial.
3. **Almacenamiento**: El vector se almacena junto con el texto original en la base de datos.
4. **Consulta**: Las consultas también se vectorizan, y se comparan con los vectores almacenados utilizando métricas de similitud como la distancia coseno.
