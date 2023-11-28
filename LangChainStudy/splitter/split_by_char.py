# This is a long document we can split up.
with open('./LangChainStudy/demo_text.txt') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])