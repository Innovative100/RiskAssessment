# %%
# pip install --upgrade langchain
# pip install pypdf
# pip install faiss-cpu

# %%
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import AzureChatOpenAI

# %%
model_llm = AzureChatOpenAI(
    temperature = 0.1,
    azure_deployment = 'deployment-text-risk-assessment',
    # azure_endpoint = os.getenv(),
    # openai_api_type = os.getenv()
    openai_api_type = 'azure',
    openai_api_version = '2023-08-01-preview', 
    openai_api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_base = 'https://instance-text-risk-assessment.openai.azure.com/',
)

# %% [markdown]
# ### Define Tool #1 - BingSearch 

# %%
from langchain.utilities import BingSearchAPIWrapper
from langchain.tools.bing_search.tool import BingSearchRun
from langchain.agents import AgentType, initialize_agent, Tool

# os.environ['BING_SUBSCRIPTION_KEY']=os.getenv("BING_SUBSCRIPTION_KEY")
os.environ['BING_SUBSCRIPTION_KEY']= "28f56faa9d3245648b585190be684fdb"
# os.environ['BING_SUBSCRIPTION_KEY']= os.getenv('BING_SUBSCRIPTION_KEY')
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

searchWrapper = BingSearchAPIWrapper()
tool_BingSearch = BingSearchRun(api_wrapper=searchWrapper)


query_websearch = "what is django"
query_websearch = "what is api gateway"
# result = tool_BingSearch(query_websearch)
# print(result)

# %% [markdown]
# #### Build Knowlege Base - Step 1: upload & split documents

# %%
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

_= """
filepath = ('./data/risk_assessment')
loader = PyPDFDirectoryLoader(filepath)

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents_chunks = text_splitter.split_documents(documents_original)

documents_chunks

"""

# %% [markdown]
# #### Build Knowledge Base - Step 2: embedding for splitted documents, store text chunks & embeddings into vector db.

# %%
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings

model_embedding = OpenAIEmbeddings(
    deployment = 'deployment-embedding-ada-002',
    # model = ,
    chunk_size = 16,
    openai_api_type = 'azure',
    openai_api_version = '2023-08-01-preview',
    openai_api_base = 'https://instance-text-risk-assessment.openai.azure.com/',
    openai_api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    # openai_api_key = "d7078cc13f3e46138215166baa8bced4",
)

_="""
db_vector = FAISS.from_documents(
    docs_chunks,
    embedding = model_embedding,
    
)

"""
# retriever = db_vector.as_retriever()

query = 'what is total profit?'
# output = db_vector.similarity_search(query)
# print(output[0].page_content)

# db_vector.save_local("index_faiss_001")
db_vector_new = FAISS.load_local("index_faiss_001", model_embedding)
query = 'any legal issue?' 
# output = db_vector_new.similarity_search(query)
# print(output[0].page_content)


# %% [markdown]
# ### Define Tool # 2 - RAG

# %%
from langchain.tools.vectorstore.tool import VectorStoreQAWithSourcesTool

tool_RAG = VectorStoreQAWithSourcesTool(
    name = "query_tool_RAG",
    description = "",
    vectorstore = db_vector_new,
    llm = model_llm,
    verbose = True,
)

query = "what is major legal issue?"
# output = tool_RAG(query)
# print(output)

# %%
# retriever

# %% [markdown]
# #### Define Tool_set

# %%
tool_set = [
    Tool.from_function(
        name = "Search from knowledge-base",
        func = tool_RAG.run,
        description = "answer questions based on knowledge-base (i.e. collected documents ) - RAG"
    ),
    Tool.from_function(
        name = "Bing Search",
        func = tool_BingSearch.run,
        description = "answer questions based on web-search"
    ),  
]

# %% [markdown]
# ### Build Agent - without memory

# %%

agent_without_memory = initialize_agent(
    tools = tool_set,
    llm = model_llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    handle_parsing_errors = True,
)

query = "what is walmart's law suit 8 years ago?"
# agent_without_memory.run(query)

# %% [markdown]
# ### Build Agent - with memory

# %%
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain

prefix = """You are a helpful assistant.
"""

suffix = """Begin to generate result. You need to work as a helpful financial assistant. 
{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ZeroShotAgent.create_prompt(
    tools = tool_set,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["input", "chat_history", "agent_scratchpad"],
)

memory = ConversationBufferMemory(
    memory_key = "chat_history",
    
    )

chain_llm = LLMChain(
    llm = model_llm,
    prompt = prompt,
)

agent = ZeroShotAgent(
    llm_chain = chain_llm,
    tools = tool_set,
    verbose = True,
)

agent_with_memory = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tool_set,
    verbose = True,
    memory = memory,
    handle_parsing_errors = True,
)

query = "what is walmarts' law suit 5 years ago?"
# agent_with_memory.run(input = query)

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ### Define Tool - CSV Search

# %%
# pip install langchain_experimental

# _= """

from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain.agents.agent_toolkits import create_csv_agent
# from langchain.agents import create_csv_agent
from langchain.agents import AgentType, initialize_agent

agent_csv = create_csv_agent(
    model_llm,
    './data/csv/bank_portugal.csv',
    verbose = True,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

query = 'how many row are there?'
agent_csv.run(query)

# """




# %% [markdown]
# #### Build database: Store .csv files into database 

# %%
# pip install sqlite3
import sqlite3
import pandas as pd
_= """
df = pd.read_csv("d:/00_data/data_bank_account/card.csv")
df.columns = df.columns.str.strip()
df.to_sql(
    'table_card',
    sqlite3.connect('bank_account_db.db'),
    if_exists = 'replace'
)

df = pd.read_csv("d:/00_data/data_bank_account/order.csv")
df.columns = df.columns.str.strip()
df.to_sql(
    'table_order',
    sqlite3.connect('bank_account_db.db'),
    if_exists = 'replace'
)

df = pd.read_csv("d:/00_data/data_bank_account/accounts.csv")
df.columns = df.columns.str.strip()
df.to_sql(
    'table_accounts',
    sqlite3.connect('bank_account_db.db'),
    if_exists = 'replace'
)

df = pd.read_csv("d:/00_data/data_bank_account/district.csv")
df.columns = df.columns.str.strip()
df.to_sql(
    'table_district',
    sqlite3.connect('bank_account_db.db'),
    if_exists = 'replace'
)

"""

_= """
df = pd.read_csv("d:/00_data/data_bank_account/trans.csv")
df.columns = df.columns.str.strip()
df.to_sql(
    'table_trans',
    sqlite3.connect('bank_account_db.db'),
    if_exists = 'replace'
)


df_clients = pd.read_csv("d:/00_data/data_bank_account/clients.csv")
df_clients.columns = df_clients.columns.str.strip()
df_clients.to_sql(
    'table_clients',
    sqlite3.connect('./data/bank_account/bank_account_db.db'),
    if_exists = 'replace'
)

df_loans = pd.read_csv("d:/00_data/data_bank_account/loans.csv")
df_loans.columns = df_loans.columns.str.strip()
df_loans.to_sql(
    'table_loans',
    sqlite3.connect('./data/bank_account/bank_account_db.db'),
    if_exists = 'replace'
)

"""


# %% [markdown]
# #### Build database: Test database for the existance of dataset 

# %%
_= """

curr = sqlite3.connect('./data/bank_account/bank_account_db.db').cursor()

curr.execute('''SELECT * from table_clients''')
for record in curr.fetchall():
    print(record)

"""

# %% [markdown]
# ### Define Tool - SQL query 

# %%
# _= """

from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///./data/bank_account/bank_account_db.db")
tool_sql = SQLDatabaseToolkit(
    db = db,
    llm = model_llm,
)

agent_sql = create_sql_agent(
    llm = model_llm,
    toolkit = tool_sql,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
)

query = ("how many rows are there in the accounts table?")
# output = agent.run(query)
# print(output)

# """



# %%


# %%


# %%


# %%


# %%


# %%
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

_= """

template = """

# Answer the question based only on the following context: 
# {context}

# Question:{question}
_= """
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_llm
    | StrOutputParser()
)

chain.invoke("what is the profit")

"""

# %%


# %%
# pip install watermark

# %load_ext watermark
# %watermark -a "Sudarshan Koirala" -vmp langchain,openai

# %%
# pip install sqlalchemy
_= """
import sqlalchemy
df = pd.read_sql_table(
    "table_loans", 
    sqlalchemy.create_engine("sqlite:///./data/bank_account/bank_account_db.db"),
    )
"""

# %%
_= """
from langchain.chains import create_sql_query_chain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db_sql = SQLDatabase.from_uri("sqlite:///./data/bank_account/card.db")

chain_sql = create_sql_query_chain(model_llm, db_sql)

prompt = {"question":"how many rows are there?"}
output = chain.invoke(prompt)

"""
# print(output)


import streamlit as st

st.set_page_config(
    page_title = "Multiple APP"
)



st.title('Interactive Question Answer for: ')

st.write("  ")
st.write(" " + "  (1) company risk-assessment related questions (unstructured data source)")
st.write("  (2) company database related questions (structured data source - .csv data file and data tables in database)")
st.write(" ")
st.write(" ")
st.write(" ")

st.write("* Tool 1: Query-interface for risk-assessment related questions")
user_web_input = st.text_input('Enter you company risk-factor related query below:')
if st.button('Click to submit query (risk assessment related)') and user_web_input:
    output = agent_without_memory.run(user_web_input)
    st.write(output)

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write("* Tool 2.1: Query-interface for database related questions - .csv data file")
user_web_input = st.text_input('Enter you database related query below:')
if st.button('Click to submit query (.csv file related)') and user_web_input:
    output = agent_csv.run(user_web_input)
    st.write(output)

st.write(" ")
st.write(" ")
st.write(" ")

st.write("* Tool 2.2: Query-interface for database related questions - data tables in database")
user_web_input = st.text_input('Enter you database related query below: ')
if st.button('Click to submit query (database related)') and user_web_input:
    output = agent_sql.run(user_web_input)
    st.write(output)

# st.sidebar.success("Select a page above")


