from flask import Flask, render_template, request, session
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
import os

app = Flask(__name__)
app.secret_key = "HiSanket" # Required for session-based memory

# 1. Initialize Groq via LangChain
os.environ["GROQ_API_KEY"] = ""
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


chat_sessions = {}

def get_chat_chain(session_id):
    if session_id not in chat_sessions:
        # Create a new memory object for this specific user session
        memory = ConversationBufferMemory()
        chat_sessions[session_id] = ConversationChain(llm=llm, memory=memory)
    return chat_sessions[session_id]

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("msg")
    # Use a session ID to keep users' conversations separate
    session_id = session.get("user_id", "default_user") 
    
    chain = get_chat_chain(session_id)
    response = chain.predict(input=user_msg)
    
    return {"response": response}

if __name__ == "__main__":
    app.run(debug=True)