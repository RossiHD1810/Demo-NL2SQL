import os
os.environ['OPENai_API_KEY']='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'
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core. runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from transformers import T5ForConditionalGeneration, T5Tokenizer

model=T5ForConditionalGeneration, T5Tokenizer
tokenizer=T5Tokenizer.from_pretrained('t5-small')

#Funzione per Generare il Testo
def generate_text(prompt_text):
    input_ids=tokenizer.encode(prompt_text, return_tensor='pt')
    outputs=model.generate(input_ids, max_lenght=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_schema(_):
    return db.get_table_info()

template = f""""
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query :
"""

db_uri="mysql+mysqlconnector://root:rossimirco1810@localhost:3306/chinook"
db=SQLDatabase.from_uri(db_uri)
""" print(db.run('SELECT * FROM Album LIMIT 5')) """

result = generate_text(prompt_text)
print(result)




#sql_chain.invoke({"question":"how many artists are there?"})