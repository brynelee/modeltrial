from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

# Full list of support languages
print([e.value for e in Language])

print("=" * 50)
print("=" * 50)

# You can also see the separators used for a given language
chars = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
print(chars)

print("=" * 50)
print("=" * 50)

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)


print("=" * 50)
print("=" * 50)


JS_CODE = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
print(js_docs)


print("=" * 50)
print("=" * 50)

TS_CODE = """
function helloWorld(): void {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

ts_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS, chunk_size=60, chunk_overlap=0
)
ts_docs = ts_splitter.create_documents([TS_CODE])
print(ts_docs)


print("=" * 50)
print("=" * 50)

markdown_text = """
# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

```bash
# Hopefully this code block isn't split
pip install langchain
```

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text])
print(md_docs)