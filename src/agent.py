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

# Maximum number of messages to keep in context to avoid token overflow
MAX_HISTORY_MESSAGES = 10


class AgentState(TypedDict):
    """State schema for ReAct agent."""

    messages: Annotated[Sequence[BaseMessage], add]
    retrieved_context: str
    iteration_count: int


class ReActAgent:
    """LangGraph ReAct agent for document Q&A."""

    def __init__(self, retrieval_service: RetrievalService, tools: list = None):
        """
        Initialize ReAct agent.

        Args:
            retrieval_service: Retrieval service instance
            tools: Optional list of tools (will use retrieval service if not provided)
        """
        self.retrieval_service = retrieval_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
        )

        # Bind tools to LLM if provided
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            # Create a simple tool that uses retrieval service
            self.llm_with_tools = self.llm

        # Build the graph
        self.graph = self._build_graph(tools)

        # Compile without checkpointer - we manage chat history in Streamlit session_state
        # This prevents token overflow from accumulated messages
        self.app = self.graph.compile()

    def _build_graph(self, tools: list = None) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("think", self._think_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Add tool node if tools are provided
        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.set_entry_point("think")

        # Add edges
        workflow.add_edge("think", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Add conditional edge for tools if available
        if tools:
            # This would need conditional logic to decide when to use tools
            pass

        return workflow

    def _trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim messages to prevent context overflow.
        Keeps the most recent messages up to MAX_HISTORY_MESSAGES.
        """
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages
        
        # Keep the most recent messages
        trimmed = messages[-MAX_HISTORY_MESSAGES:]
        logger.info(f"Trimmed message history from {len(messages)} to {len(trimmed)} messages")
        return trimmed

    def _think_node(self, state: AgentState) -> AgentState:
        """Think node: Analyze the query and decide what to retrieve."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if isinstance(last_message, HumanMessage):
            # Add system prompt for reasoning
            system_msg = SystemMessage(content=REACT_SYSTEM_PROMPT)
            messages_with_system = [system_msg] + list(messages)

            # Get LLM reasoning (simplified - in full ReAct, this would be more complex)
            response = self.llm_with_tools.invoke(messages_with_system)
            state["messages"] = list(messages) + [response]

        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve node: Get relevant context from documents."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if isinstance(last_message, (HumanMessage, AIMessage)):
            # Extract query from last message
            query = last_message.content if isinstance(last_message, HumanMessage) else None

            if not query and isinstance(last_message, AIMessage):
                # Try to extract query from AI message (if it contains tool calls)
                query = str(last_message.content)

            if query:
                # Retrieve relevant context
                results = self.retrieval_service.retrieve(query=query)
                context = self.retrieval_service.get_context_text(results)
                state["retrieved_context"] = context
                logger.info(f"Retrieved context with {len(results)} chunks")

        return state

    def _generate_node(self, state: AgentState) -> AgentState:
        """Generate node: Create final response with context."""
        messages = state.get("messages", [])
        retrieved_context = state.get("retrieved_context", "")
        iteration_count = state.get("iteration_count", 0) + 1

        # Get user question (last human message)
        user_question = None
        conversation_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
            elif isinstance(msg, AIMessage):
                conversation_history.append(f"Assistant: {msg.content}")

        if user_question:
            # Format conversation history
            history_text = "\n".join(conversation_history[-3:]) if conversation_history else "None"

            # Create prompt with context
            prompt = FINAL_RESPONSE_PROMPT.format(
                context=retrieved_context,
                conversation_history=history_text,
                question=user_question,
            )

            # Generate response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["messages"] = list(messages) + [response]
            state["iteration_count"] = iteration_count

        return state

    def invoke(self, query: str, config: dict = None, chat_history: List[BaseMessage] = None) -> str:
        """
        Invoke the agent with a query.

        Args:
            query: User query
            config: Optional configuration (not used, kept for compatibility)
            chat_history: Optional list of previous messages for context

        Returns:
            Agent response
        """
        # Build initial messages with optional chat history
        messages = []
        if chat_history:
            # Trim chat history to avoid context overflow
            messages = self._trim_messages(chat_history)
        
        # Add current query
        messages.append(HumanMessage(content=query))
        
        initial_state = {
            "messages": messages,
            "retrieved_context": "",
            "iteration_count": 0,
        }

        result = self.app.invoke(initial_state)

        # Extract final response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return last_message.content

        return "I apologize, but I couldn't generate a response."
