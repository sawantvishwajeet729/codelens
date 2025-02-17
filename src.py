#importing libraries
import os
import sys
import re
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import rank_bm25
import streamlit as st

#git_token = st.secrets['git_token']
git_token = "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"

headers = {
    "Authorization": f"Bearer {git_token}",
    "Accept": "application/vnd.github.v3+json"
}

#extract the owner and repo name from the repo link
def extract_repo_info(url):
    """
    Extracts the owner and repo name from a GitHub URL.
    Expects URL in the form: https://github.com/<owner>/<repo>
    """
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        owner, repo = match.group(1), match.group(2)
        # Remove .git suffix if present
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo
    else:
        raise ValueError("Invalid GitHub repository URL.")
    



#identify if the file is binary or not. If it is binary, skip it. Since binary files cannot be added as text
def is_binary_extension(filename):
    """
    Check if a file is binary based on its extension.
    """
    # List of file extensions we want to ignore (binary files)
    BINARY_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
                     '.pdf', '.zip', '.tar', '.gz', '.mp3', '.wav',
                     '.ogg', '.mp4', '.avi'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in BINARY_EXTENSIONS


#get the text format of the file from the GitHub URL
def get_text_from_url(download_url):
    try:
        response = requests.get(download_url, headers=headers)
        response.raise_for_status()
        # Try decoding as UTF-8. If it fails, we assume it's not a text file.
        return response.content.decode('utf-8')
    except UnicodeDecodeError:
        #print(f"Skipping (non-text or binary): {download_url}")
        return None
    except Exception as e:
        #print(f"Error fetching {download_url}: {e}")
        return None
    


#process the directory and get the text from the files
def process_directory(owner, repo, path=""):
    """Recursively process the directory at the given path."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Error accessing {api_url}: {response.status_code}")
        return []
    
    items = response.json()
    texts = []
    
    # The API returns a list when path is a directory.
    for item in items:
        item_type = item.get('type')
        if item_type == 'dir':
            # Recurse into subdirectories.
            texts.extend(process_directory(owner, repo, item.get('path')))
        elif item_type == 'file':
            filename = item.get('name')
            # Skip files with known binary extensions.
            if is_binary_extension(filename):
                #print(f"Skipping binary file (by extension): {item.get('path')}")
                continue

            download_url = item.get('download_url')
            if download_url:
                file_text = get_text_from_url(download_url)
                if file_text is not None:
                    texts.append(
                        f"----- Start of file: {item.get('path')} -----\n"
                        f"{file_text}\n"
                        f"----- End of file: {item.get('path')} -----\n\n"
                    )
    return texts


#create vector embedding of the combined text of the repository

def vector_embdeddings(all_text):
    #Text spliiter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n# "],
        chunk_size=1200,
        chunk_overlap=250
    )
    chunks = text_splitter.split_text(all_text)

    # Create vector store from text chunks
    db = FAISS.from_texts(chunks, OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536))

    # Vector retriever from vector store
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
    # Keyword retriever
    keyword_retriever = BM25Retriever.from_texts(chunks)
    # Hybrid retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever
    