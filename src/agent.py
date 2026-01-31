# =============================================================================
# LANGGRAPH REACT AGENT - The "Brain" of the RAG Chat Assistant
# =============================================================================
#
# This module implements the core AI agent using LangGraph's ReAct pattern.
# ReAct stands for "Reasoning and Acting" - the agent thinks about what to do,
# takes actions (like searching documents), observes results, and generates answers.
#
# Architecture Flow:
#     User Question --> Think Node --> Retrieve Node --> Generate Node --> Answer
#
# Why LangGraph?
#     - Built-in state management (tracks conversation context)
#     - Easy to define multi-step reasoning workflows
#     - Integrates well with LangChain tools
#
# Key Design Decisions:
#     1. No MemorySaver checkpointer - We manage chat history in Streamlit's
#        session_state instead. This prevents unbounded memory growth.
#     2. Message trimming to last 10 messages - Prevents exceeding OpenAI's
#        128K token context limit.
#     3. Simple linear flow (think -> retrieve -> generate) - More complex
#        ReAct loops with tool calling can be added later.
#
# =============================================================================

"""LangGraph ReAct agent implementation."""

from typing import TypedDict, Annotated, Sequence, List
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.retrieval import RetrievalService
from src.prompts import REACT_SYSTEM_PROMPT, FINAL_RESPONSE_PROMPT
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# IMPORTANT: This limit prevents the "context_length_exceeded" error from OpenAI.
# GPT-4-turbo has a 128K token limit. Each message can be ~1000+ tokens.
# Keeping only the last 10 messages ensures we stay well under the limit.
# -----------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 10


class AgentState(TypedDict):
    """
    State schema for the ReAct agent.
    
    This is the "memory" that flows through the LangGraph nodes.
    Each node can read from and write to this state.
    
    Fields:
        messages: List of conversation messages (human and AI). Uses the 'add'
                  operator to append new messages rather than replace.
        retrieved_context: Document chunks found by the retrieval system.
                          This gets injected into the final prompt.
        iteration_count: Tracks how many reasoning loops we've done.
                        Useful for debugging and preventing infinite loops.
    """
    messages: Annotated[Sequence[BaseMessage], add]
    retrieved_context: str
    iteration_count: int


class ReActAgent:
    """
    LangGraph ReAct agent for document Q&A.
    
    This is the main class that orchestrates the entire question-answering flow:
    1. Takes a user question
    2. Retrieves relevant document chunks from the vector store
    3. Uses GPT-4 to generate an answer based on the retrieved context
    
    The agent uses a state machine pattern where each "node" is a processing step.
    
    Example Usage:
        retrieval_service = RetrievalService(vector_store)
        agent = ReActAgent(retrieval_service)
        answer = agent.invoke("What is the person's experience?")
    """

    def __init__(self, retrieval_service: RetrievalService, tools: list = None):
        """
        Initialize the ReAct agent.

        Args:
            retrieval_service: The service that fetches relevant document chunks.
                              This connects to the Chroma vector database.
            tools: Optional MCP tools for advanced use cases. If not provided,
                   the agent uses the retrieval_service directly.
        """
        self.retrieval_service = retrieval_service
        
        # Initialize the OpenAI LLM (Large Language Model)
        # This is what generates the actual responses
        self.llm = ChatOpenAI(
            model=settings.openai_model,          # e.g., "gpt-4-turbo-preview"
            temperature=settings.temperature,      # 0.7 = balanced creativity
            api_key=settings.openai_api_key,
        )

        # If MCP tools are provided, bind them to the LLM
        # This allows the LLM to call tools like search_documents
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            # No tools - agent will use retrieval service directly
            self.llm_with_tools = self.llm

        # Build the LangGraph state machine
        self.graph = self._build_graph(tools)

        # Compile the graph into an executable application
        # NOTE: We intentionally do NOT use a checkpointer here.
        # The checkpointer would accumulate all messages forever, causing
        # the "context_length_exceeded" error we fixed earlier.
        self.app = self.graph.compile()

    def _build_graph(self, tools: list = None) -> StateGraph:
        """
        Build the LangGraph workflow (state machine).
        
        This defines the processing pipeline:
        
            START --> think --> retrieve --> generate --> END
        
        Each node is a function that transforms the agent state.
        
        Args:
            tools: Optional MCP tools (for future tool-calling support)
            
        Returns:
            A configured StateGraph ready to be compiled
        """
        # Create a new state graph with our AgentState schema
        workflow = StateGraph(AgentState)

        # ----- Add processing nodes -----
        # Each node is a function that takes state and returns updated state
        
        workflow.add_node("think", self._think_node)       # Analyze the question
        workflow.add_node("retrieve", self._retrieve_node) # Fetch relevant docs
        workflow.add_node("generate", self._generate_node) # Create the answer

        # Add tool node if MCP tools are provided (for advanced use cases)
        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)

        # ----- Define the flow -----
        # This is a simple linear flow: think -> retrieve -> generate -> done
        workflow.set_entry_point("think")
        workflow.add_edge("think", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Future enhancement: Add conditional edges for tool calling
        # This would allow the agent to decide whether to use tools
        if tools:
            pass  # TODO: Implement conditional tool routing

        return workflow

    def _trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim the message history to prevent context overflow.
        
        WHY THIS IS IMPORTANT:
        OpenAI's GPT-4-turbo has a 128,000 token limit. If we send too many
        messages, we get the "context_length_exceeded" error. By keeping only
        the most recent messages, we ensure the conversation stays within limits.
        
        Args:
            messages: Full list of conversation messages
            
        Returns:
            Trimmed list with at most MAX_HISTORY_MESSAGES (10) messages
        """
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages
        
        # Keep only the most recent messages
        # This means older context is lost, but prevents crashes
        trimmed = messages[-MAX_HISTORY_MESSAGES:]
        logger.info(f"Trimmed message history from {len(messages)} to {len(trimmed)} messages")
        return trimmed

    def _think_node(self, state: AgentState) -> AgentState:
        """
        Think Node: Analyze the user's question and prepare for retrieval.
        
        This is the first step in the ReAct loop. The LLM looks at the question
        and thinks about what information it needs to answer it.
        
        In a more advanced implementation, this node would:
        - Decompose complex questions into sub-questions
        - Decide which tools to use
        - Plan a multi-step retrieval strategy
        
        Current implementation: Simply passes the message through with a system prompt.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Updated state with LLM's reasoning added to messages
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        # Only process if the last message is from a human (user)
        if isinstance(last_message, HumanMessage):
            # Add the system prompt that tells the LLM how to behave
            system_msg = SystemMessage(content=REACT_SYSTEM_PROMPT)
            messages_with_system = [system_msg] + list(messages)

            # Get the LLM's reasoning about how to answer
            response = self.llm_with_tools.invoke(messages_with_system)
            
            # Add the LLM's response to the message history
            state["messages"] = list(messages) + [response]

        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """
        Retrieve Node: Fetch relevant document chunks from the vector store.
        
        This is where the RAG (Retrieval-Augmented Generation) magic happens:
        1. Extract the query from the conversation
        2. Search the Chroma vector database for similar content
        3. Store the retrieved chunks in the state for the generate node
        
        The retrieval service uses semantic search - it finds documents that
        are conceptually similar to the query, not just keyword matches.
        
        Args:
            state: Current agent state
            
        Returns:
            State with retrieved_context populated with relevant chunks
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if isinstance(last_message, (HumanMessage, AIMessage)):
            # Extract the query text from the message
            # For human messages, use the content directly
            # For AI messages, this might contain tool call info
            query = last_message.content if isinstance(last_message, HumanMessage) else None

            if not query and isinstance(last_message, AIMessage):
                # Fallback: use AI message content as query
                query = str(last_message.content)

            if query:
                # Search the vector store for relevant document chunks
                # This returns chunks sorted by semantic similarity
                results = self.retrieval_service.retrieve(query=query)
                
                # Format the results into a context string
                # This will be injected into the prompt for the LLM
                context = self.retrieval_service.get_context_text(results)
                state["retrieved_context"] = context
                
                logger.info(f"Retrieved context with {len(results)} chunks")

        return state

    def _generate_node(self, state: AgentState) -> AgentState:
        """
        Generate Node: Create the final answer using the retrieved context.
        
        This is the final step where we:
        1. Take the user's question
        2. Combine it with the retrieved document chunks
        3. Ask the LLM to generate an answer based on this context
        
        The prompt instructs the LLM to:
        - Only use information from the provided context
        - Cite sources (document name and chunk index)
        - Admit when it doesn't have the information
        
        Args:
            state: State containing messages and retrieved_context
            
        Returns:
            State with the final AI response added to messages
        """
        messages = state.get("messages", [])
        retrieved_context = state.get("retrieved_context", "")
        iteration_count = state.get("iteration_count", 0) + 1

        # Find the user's original question (last human message)
        user_question = None
        conversation_history = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
            elif isinstance(msg, AIMessage):
                # Keep track of previous AI responses for context
                conversation_history.append(f"Assistant: {msg.content}")

        if user_question:
            # Format recent conversation history (last 3 exchanges)
            history_text = "\n".join(conversation_history[-3:]) if conversation_history else "None"

            # Build the final prompt with context injection
            # This is the key RAG step - we give the LLM the retrieved documents
            prompt = FINAL_RESPONSE_PROMPT.format(
                context=retrieved_context,           # Document chunks go here
                conversation_history=history_text,   # Recent conversation
                question=user_question,              # The user's question
            )

            # Generate the final response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Update state with the response
            state["messages"] = list(messages) + [response]
            state["iteration_count"] = iteration_count

        return state

    def invoke(self, query: str, config: dict = None, chat_history: List[BaseMessage] = None) -> str:
        """
        Main entry point: Run the agent with a user query.
        
        This is the method you call to get an answer from the agent.
        It orchestrates the entire flow: think -> retrieve -> generate.
        
        Args:
            query: The user's question (e.g., "What is this person's experience?")
            config: Optional configuration (kept for API compatibility, not used)
            chat_history: Optional previous messages for context. These are
                         trimmed to prevent token overflow.
        
        Returns:
            The agent's answer as a string, with source citations
        
        Example:
            response = agent.invoke("What college did they attend?")
            print(response)  # "Based on the document, they attended..."
        """
        # Start with any provided chat history
        messages = []
        if chat_history:
            # IMPORTANT: Trim history to prevent context overflow
            messages = self._trim_messages(chat_history)
        
        # Add the current user query
        messages.append(HumanMessage(content=query))
        
        # Initialize the agent state
        initial_state = {
            "messages": messages,
            "retrieved_context": "",
            "iteration_count": 0,
        }

        # Run the LangGraph workflow
        # This executes: think -> retrieve -> generate
        result = self.app.invoke(initial_state)

        # Extract the final response from the result
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return last_message.content

        # Fallback if something went wrong
        return "I apologize, but I couldn't generate a response."
