# This is a long document we can split up.
with open('./LangChainStudy/demo_text.txt') as f:
    state_of_the_union = f.read()
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)

""" for each in texts:
    print(each) """

print(texts[0])

print("=" * 80)
print("=" * 80)

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])

print("=" * 80)
print("=" * 80)


""" from langchain.text_splitter import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=1000)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0]) """

import nltk

nltk.download('punkt')

from langchain.text_splitter import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=1000)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])



