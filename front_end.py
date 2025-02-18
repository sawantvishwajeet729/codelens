import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src import *

#set API key
os.environ['OPENAI_API_KEY']= st.secrets["openAIKey"]
groq_key=st.secrets["groqKey"]
git_token = st.secrets['git_token']

headers = {
    "Authorization": f"Bearer {git_token}",
    "Accept": "application/vnd.github.v3+json"
}

#set up the model
#initialise the langchain chat prompt template
system_prompt = """You are an AI assistant designed to answer questions based on the contents of a given text file. The text file is a structured combination of all textual files from a GitHub repository. 
Your primary goal is to provide accurate, concise, and helpful answers based on the provided text file. If a question goes beyond the file’s content, acknowledge the limitation and avoid making assumptions. Follow these guidelines:
    Stay Within Context: Only use information available in the provided text file. If an answer requires external knowledge, state that you can’t provide an answer.
    Be Precise & Concise: Give direct, relevant answers. If necessary, provide file references or specific sections to support your response.
    Understand Code & Documentation: If the file contains code, explain it clearly, including functionality, dependencies, and usage when relevant.
    Clarify Ambiguities: If the question is unclear or too broad, ask for clarification rather than guessing.
    Handle Errors Gracefully: If the text file is incomplete or corrupted, notify the user about the limitation and provide the best response possible based on the available data.*
When responding, maintain a professional yet approachable tone. Your purpose is to assist users in understanding the contents of the repository efficiently."""
user_prompt = "Question: {input}\n\nRelevant Context:\n{context}"
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])

# Initialize the llm model
model = ChatGroq(groq_api_key=groq_key,
                 model="llama-3.3-70b-versatile",
                 temperature=0,
                 max_tokens=None,
                 timeout=None,
                 max_retries=2)

output_parser=StrOutputParser()

#create document chain
document_chain = create_stuff_documents_chain(model, prompt)


# Set page configuration
st.set_page_config(page_title="CodeLens", page_icon="🤖", layout="centered")

# create 3 tabs for home, about and donate buttons
tab1, tab2, tab3 = st.tabs(["Home", "About", "Donate"])

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.2rem;
    font-weight:normal;
    }

     /* Make text of selected tab bold */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        font-weight: bold !important;
        color: #ef476f !important;
    }

    /* Background color */
    .stApp {
        background-color: #f5f5dc;
    }

    .navbar .logo {
        font-size: 22px;
        font-weight: bold;
        color: white;
    }

    .navbar .nav-links a {
        color: white;
        text-decoration: none;
        font-size: 18px;
        padding: 14px 20px;
        display: inline-block;
        transition: background 0.3s;
    }

    .navbar .nav-links a:hover {
        background: #b56576;
        border-radius: 5px;
    }

    /* Add space at top to avoid content being hidden */
    .spacer {
        height: 60px;
    }

    /* Animated title */
    @keyframes colorChange {
        0% { color: #d4a373; }
        50% { color: #b56576; }
        100% { color: #d4a373; }
    }
    
    .title {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        animation: colorChange 3s infinite alternate;
    }

    /* Style buttons */
    .stButton>button {
        background-color: #ef476f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        transition: transform 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        color:white;
        background-color: #C8385A;
    }

    /* Style input fields */
    input, textarea {
        border: 1px solid #d4a373;
        border-radius: 5px;
        padding: 8px;
    }
    </style>

    <script>
    function scrollToSection(section) {
        document.getElementById(section).scrollIntoView({ behavior: 'smooth' });
    }
    </script>
    """,
    unsafe_allow_html=True,
)
with tab1:    

    # Home Section
    st.markdown('<div id="home"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title">🔍 CodeLens</h1>', unsafe_allow_html=True)
    st.write(
        """
        **CodeLens** is an AI-powered web application that extracts all non-binary files from a GitHub repository, combines them into a single text document, and converts the content into a vector database using OpenAI embeddings and FAISS. Users can then query the repository’s knowledge using a powerful LLM, enabling instant insights and answers. The app also provides an option to download the combined text file for further analysis.
        """
    )

    # Process Repository Section
    st.header("⏳ Process Repository")
    repo_url = st.text_input("Enter GitHub Repository URL:")

    if st.button("Process Repository"):
        if repo_url:
            # Clear session state
            st.session_state.clear()

            #get the repository files and combine
            owner, repo = extract_repo_info(repo_url)
            all_texts = process_directory(owner, repo)
            all_texts = "\n\n".join(all_texts)
            st.session_state['all_texts'] = all_texts
            st.success("Repository processed successfully!")

            #downloading and viewing the combined text
            with st.expander("View the Combined code"):
                # Define scrollable container using markdown and CSS
                st.text_area(f"The combined code for {repo_url}", all_texts, height=300)
        else:
            st.error("Please enter a valid GitHub repository URL.")

        #add empty line for spacing
        st.write("")

        # Add a download button
        st.download_button(
            label="Download as .txt file",
            data=all_texts,
            file_name="generated_text.txt",
            mime="text/plain"
        )

    # Ask a Question Section
    st.header("💡 Ask a Question")
    user_question = st.text_area("Enter your question:")

    if st.button("Get Answer"):
        if not user_question:
            st.error("Please enter your question.")
        else:
            if "embeddings" not in st.session_state or not st.session_state["embeddings"]:
                st.session_state["embeddings"] = vector_embdeddings(st.session_state['all_texts'])
                st.warning('New embedding was created', icon="⚠️")
                
            retrival_chain = create_retrieval_chain(st.session_state["embeddings"], document_chain)
            response = retrival_chain.invoke({'input': user_question})
            st.subheader("CodeLens Response:")
            st.text_area("Reponse:", response['answer'], height=300)

with tab2:
    # About Section
    st.markdown('<div id="about"></div>', unsafe_allow_html=True)
    st.header("📜 About CodeLens")
    st.write(
        """
        **CodeLens** is an AI-powered tool that helps developers extract insights from their GitHub repositories.  
        - **How It Works:**  
        1. Extracts text from non-binary files in a GitHub repo  
        2. Converts text into vectors using OpenAI embeddings and FAISS  
        3. Allows users to query an LLM for intelligent answers  

        Whether you're analyzing documentation, searching for code snippets, or just exploring a repo's content, **CodeLens** makes it easy! 🚀
        """
    )
    with st.container():
        st.header('✉️ Connect with me')
        st.write("[Email >](sawantvishwajeet729@gmail.com)")
        st.write("[LinkedIn >](https://www.linkedin.com/in/sawantvishwajeet729/)")
        st.write("[Medium >](https://medium.com/@sawantvishwajeet729)")
        st.write("[Github >](https://github.com/sawantvishwajeet729)")


with tab3:
    st.markdown('<div id="Donate"></div>', unsafe_allow_html=True)
    st.header("☕ Donate")
    st.write (
        """
        Hey there! 😊 Love CodeLens ? 🚀 Support me to keep it running and making new free web apps! Every little bit helps. 💙 Thank you!
        
        """
    )
    st.write("")
    st.markdown(
    """
    <style>
    .donate-button {
        background-color: #ef476f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 16px;
        transition: transform 0.2s ease-in-out;
    }
    .donate-button:hover {
        transform: scale(1.05);
        color:white;
        background-color: #C8385A;
    }
    </style>
    
    <a class="donate-button" href="https://buymeacoffee.com/sawantvishwajeet729" target="_blank">
        Buy me a Coffee 🧋
    </a>
    """,
    unsafe_allow_html=True
)
    