
import os
import streamlit as st # Import Streamlit
from langchain_core.messages import HumanMessage, SystemMessage
from groq import Groq
from langchain_groq import ChatGroq

# --- Streamlit App UI ---
st.set_page_config(page_title="Simple Agentic AI Demo", layout="centered")
st.title("🤖 Super Simple Agentic AI Demo")
st.markdown("This app demonstrates how two AI Agents can collaborate to achieve a goal.")
st.markdown("---")
st.header("The Agents at Work:")


GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.0
)
def fact_finder_agent(topic: str) -> str:
   """
   Simulates a 'Fact Finder Agent' that uses an LLM to 'search' for a fact.
   In a real scenario, this would use a web search tool (e.g., Google Search API).
   """
   st.info(f"Agent 1 (Fact Finder): Receiving request to find a fact about '{topic}'...")
   prompt = [
       SystemMessage(
           content=(
               "You are a helpful 'Fact Finder Agent'. "
               "Your job is to find one interesting, concise fact about the user's topic. "
               "Act as if you have just performed a quick search. "
               "Return ONLY the fact, no extra conversation."
           )
       ),
       HumanMessage(content=f"Tell me one interesting fact about {topic}."),
   ]
   response = llm.invoke(prompt)
   fact = response.content
   st.success(f"Agent 1 (Fact Finder): Found this fact: '{fact[:70]}...'") # show truncated fact for UI
   return fact

# --- Agent 2: The Summarizer Agent ---
# Job: To take a provided piece of text (the fact from Agent 1) and summarize it very briefly.
def summarizer_agent(text_to_summarize: str) -> str:
   """
   Simulates a 'Summarizer Agent' that uses an LLM to create a very brief summary.
   It takes input from another agent.
   """
   st.info(f"Agent 2 (Summarizer): Receiving text to summarize (length {len(text_to_summarize)} chars)...")
   prompt = [
       SystemMessage(
           content=(
               "You are a helpful 'Summarizer Agent'. "
               "Your job is to take the provided text and summarize it into a single, very brief sentence. "
               "Return ONLY the summary, no extra conversation."
           )
       ),
       HumanMessage(content=f"Summarize this: {text_to_summarize}"),
   ]
   response = llm.invoke(prompt)
   summary = response.content
   st.success(f"Agent 2 (Summarizer): Created summary: '{summary}'")
   return summary


# --- Orchestration: How the Agents Work Together ---


# 1. User provides an initial input/goal via Streamlit text input
user_topic = st.text_input("Enter a topic for our AI Agents (e.g., 'solar system', 'honey bees', 'pizza'):", key="user_topic_input")


if user_topic: # Only proceed if the user has entered a topic
   st.subheader(f"User's Goal: Get a summarized fact about '{user_topic}'")
   st.write("---")


   with st.spinner("Agents are working..."):
       # 2. Agent 1 (Fact Finder) is activated with the user's topic
       found_fact = fact_finder_agent(user_topic)


       # 3. The output (observation) from Agent 1 becomes the input for Agent 2
       final_summary = summarizer_agent(found_fact)


   st.markdown("---")
   st.subheader("🎉 Agentic AI Workflow Completed!")
   st.write(f"**Original Topic:** '{user_topic}'")
   st.write(f"**Final Summarized Fact (from Agentic AI):** '{final_summary}'")
   st.markdown("This demonstrates how AI Agents can work together, passing information and completing a task step-by-step!")

