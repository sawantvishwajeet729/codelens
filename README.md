# CodeLens 
================

!This readme file was written by Codelens

## Introduction
CodeLens is an AI-powered web application that extracts all non-binary files from a GitHub repository, combines them into a single text document, and converts the content into a vector database using OpenAI embeddings and FAISS. Users can then query the repository’s knowledge using a powerful LLM, enabling instant insights and answers.

## Features
* Extracts text from non-binary files in a GitHub repository
* Combines text into a single document
* Converts text into vectors using OpenAI embeddings and FAISS
* Allows users to query the repository's knowledge using a powerful LLM
* Provides instant insights and answers
* Option to download the combined text file for further analysis

## Requirements
* Python 3.x
* Streamlit
* Langchain
* OpenAI API key
* Groq API key

## Installation
1. Clone the repository: `git clone https://github.com/your-username/CodeLens.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Set up your OpenAI API key and Groq API key in the `front_end.py` file
4. Run the application: `streamlit run front_end.py`

## Usage
1. Enter the GitHub repository URL in the input field
2. Click the "Process Repository" button to extract and combine the text
3. Enter your question in the input field
4. Click the "Get Answer" button to receive an answer from the LLM
5. Option to download the combined text file for further analysis

## About
CodeLens is an AI-powered tool that helps developers extract insights from their GitHub repositories. It uses OpenAI embeddings and FAISS to convert the text into vectors, allowing for efficient querying and answering.

## Donate
If you like CodeLens, consider supporting us to keep it running and growing! Every little bit helps.

## API Documentation
### OpenAI API
* Model: `text-embedding-3-large`
* Dimensions: `1536`

### Groq API
* Model: `llama-3.3-70b-versatile`
* Temperature: `0`
* Max tokens: `None`
* Timeout: `None`
* Max retries: `2`

## Contributing
Contributions are welcome! If you have any ideas or suggestions, please open an issue or pull request.

## License
CodeLens is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
* OpenAI for their embeddings and LLM models
* Groq for their LLM model
* Langchain for their library and tools
* Streamlit for their web application framework