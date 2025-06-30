"""
gradio_app.py - Gradio deployment for JazzBot RAG Chatbot

This script converts your Flask-based RAG chatbot to use Gradio interface
for easier deployment and better user experience.
"""

import gradio as gr
import time
import threading
from datetime import datetime
from typing import List, Tuple, Generator
import os

# Import your existing services
from config import Config
from models.memory import ConversationMemory
from services.analyzer import QueryAnalyzer, CategoryDetector
from services.prompt import PromptEngine
from services.retriever import RetrieverService
from services.translator import TranslationService
from utils.helpers import save_response
from utils.logger import setup_logger
from langchain_community.llms import Ollama

# Configure logging
logger = setup_logger(__name__)

class GradioJazzBot:
    """Gradio interface for JazzBot RAG system"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Gradio JazzBot...")
        
        # Initialize configuration
        self.config = Config()
        
        # Initialize services
        self.memory_manager = ConversationMemory()
        self.query_analyzer = QueryAnalyzer()
        self.category_detector = CategoryDetector()
        self.prompt_engine = PromptEngine()
        self.retriever_service = RetrieverService(self.config)
        self.translator_service = TranslationService()
        
        # Initialize LLM
        try:
            self.llm = Ollama(
                model=self.config.LLM_MODEL, 
                temperature=self.config.LLM_TEMPERATURE
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None
            
        # Build retriever
        try:
            self.vectorstore, self.embeddings = self.retriever_service.build_retriever()
            if self.vectorstore is None:
                logger.error("Vectorstore initialization failed")
            else:
                logger.info("Vectorstore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            self.vectorstore = None
            self.embeddings = None
    
    def chat_function(self, message: str, history: List[List[str]]) -> Generator[str, None, None]:
        """
        Main chat function for Gradio interface
        
        Args:
            message: User input message
            history: Chat history from Gradio
            
        Yields:
            Partial responses for streaming effect
        """
        try:
            user_input = message.strip()
            
            if not user_input:
                yield "Please enter a message."
                return
                
            if self.vectorstore is None:
                yield "Search functionality is currently unavailable. Please try again later."
                return
                
            if self.llm is None:
                yield "AI service is currently unavailable. Please try again later."
                return
                
            logger.info(f"Processing query: {user_input}")
            
            # Analyze query
            is_greeting = self.query_analyzer.is_greeting(user_input)
            is_roman_urdu = self.query_analyzer.is_roman_urdu(user_input)
            last_response = self.memory_manager.get_last_response()
            is_followup = self.query_analyzer.is_followup(user_input, last_response)
            detected_categories = self.category_detector.detect_categories(user_input)
            primary_category = detected_categories[0] if detected_categories else None
            
            logger.info(
                f"Query analysis - Greeting: {is_greeting}, Roman Urdu: {is_roman_urdu}, "
                f"Follow-up: {is_followup}, Categories: {detected_categories}"
            )
            
            # Handle greeting
            if is_greeting:
                response = self.prompt_engine.build_greeting_response(user_input)
                yield from self._stream_text(response, user_input, None)
                return
                
            # Search for relevant documents
            docs = []
            package_name = None
            
            try:
                if is_followup:
                    package_name = self.query_analyzer.extract_package_name(last_response)
                    search_query = (
                        f"{package_name} {primary_category or ''}".strip()
                        if package_name
                        else user_input
                    )
                else:
                    search_query = user_input
                    
                docs = self.retriever_service.search_documents(search_query, self.config.SEARCH_K)
                
                if is_followup and package_name:
                    docs = [
                        doc for doc in docs
                        if package_name.lower() in doc.page_content.lower()
                    ]
                    
            except Exception as e:
                logger.error(f"Search error: {e}")
                docs = []
                
            # Build context
            context_parts = []
            for doc in docs:
                if doc.page_content.strip():
                    metadata_info = f"[Category: {doc.metadata.get('category', 'unknown')}, Type: {doc.metadata.get('file_type', 'unknown')}]"
                    context_parts.append(f"{metadata_info} {doc.page_content}")
                    
            context = "\n\n".join(context_parts)
            memory_context = self.memory_manager.get_context_string()
            
            # Generate prompt
            prompt = self.prompt_engine.build_context_prompt_main(
                context=context,
                memory_context=memory_context,
                user_input=user_input,
                is_followup=is_followup,
                package_name=package_name,
                detected_categories=detected_categories,
            )
            
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            
            yield from self._generate_streaming_response(prompt, user_input, primary_category)
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = "I apologize, but I encountered an unexpected error. Please try again."
            yield from self._stream_text(error_msg, user_input, None)
    
    def _generate_streaming_response(self, prompt: str, user_input: str, primary_category: str) -> Generator[str, None, None]:
        """Generate streaming response from LLM"""
        full_response = []
        
        try:
            # Try streaming first
            try:
                if hasattr(self.llm, "stream"):
                    current_response = ""
                    for chunk in self.llm.stream(prompt):
                        if chunk:
                            full_response.append(chunk)
                            current_response += chunk
                            yield current_response
                            time.sleep(0.02)
                else:
                    # Fallback to regular invoke with simulated streaming
                    response = self.llm.invoke(prompt)
                    full_response.append(response)
                    
                    current_response = ""
                    words = response.split()
                    for word in words:
                        current_response += (" " if current_response else "") + word
                        yield current_response
                        time.sleep(0.08)
                        
            except Exception as stream_error:
                logger.warning(f"Streaming failed, using invoke: {stream_error}")
                response = self.llm.invoke(prompt)
                full_response.append(response)
                
                current_response = ""
                for char in response:
                    current_response += char
                    yield current_response
                    if char in [" ", ".", "!", "?", "\n"]:
                        time.sleep(0.05)
                    else:
                        time.sleep(0.01)
                        
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            error_msg = "I apologize, but I encountered an error processing your request. Please try again."
            full_response = [error_msg]
            yield error_msg
            
        # Handle memory update in background
        complete_response = "".join(full_response).strip()
        
        def update_memory():
            try:
                self.memory_manager.add_exchange(user_input, complete_response, primary_category)
                save_response(user_input, complete_response, primary_category)
            except Exception as e:
                logger.error(f"Error updating memory: {e}")
                
        threading.Thread(target=update_memory, daemon=True).start()
    
    def _stream_text(self, text: str, user_input: str, category: str) -> Generator[str, None, None]:
        """Stream text with typing effect"""
        current_text = ""
        words = text.split()
        
        for word in words:
            current_text += (" " if current_text else "") + word
            yield current_text
            time.sleep(0.05)
            
        # Update memory for greetings
        if category is None:  # This is likely a greeting
            self.memory_manager.add_exchange(user_input, text, category)
    
    def clear_memory(self) -> str:
        """Clear conversation memory"""
        try:
            self.memory_manager.clear_memory()
            return "Memory cleared successfully! 🔄"
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return "Failed to clear memory. Please try again."
    
    def get_system_status(self) -> str:
        """Get system status information"""
        try:
            memory_count = len(self.memory_manager.get_memory())
            
            status_info = f"""
            ## System Status 📊
            
            **Status:** {'✅ Online' if self.llm and self.vectorstore else '⚠️ Partial'}
            **Vectorstore:** {'✅ Available' if self.vectorstore else '❌ Unavailable'}
            **LLM:** {'✅ Available' if self.llm else '❌ Unavailable'}
            **Memory Size:** {memory_count} exchanges
            **Model:** {self.config.LLM_MODEL}
            **Temperature:** {self.config.LLM_TEMPERATURE}
            **Search K:** {self.config.SEARCH_K}
            **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return status_info.strip()
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return "Error retrieving system status."


def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Initialize the bot
    bot = GradioJazzBot()
    
    # Create custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .chat-message {
        border-radius: 10px !important;
        padding: 10px !important;
        margin: 5px 0 !important;
    }
    
    .message {
        border-radius: 10px !important;
    }
    
    footer {
        visibility: hidden;
    }
    """
    
    # Create the interface
    with gr.Blocks(
        title="JazzBot - AI Assistant",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🎵 JazzBot - AI Assistant
            
            Welcome to JazzBot! I'm here to help you with your queries using advanced RAG technology.
            Ask me anything about our services, packages, or general questions.
            """
        )
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_label=False,
                    container=True,
                    avatar_images=("🧑‍💻", "🤖")
                )
                
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    container=False,
                    scale=7,
                    show_label=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", scale=1)
            
            # Sidebar with controls and info
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                
                clear_memory_btn = gr.Button("Clear Memory 🧠", variant="secondary")
                status_btn = gr.Button("System Status 📊", variant="secondary")
                
                gr.Markdown("### Information")
                status_output = gr.Markdown("Click 'System Status' to see current status.")
                
                gr.Markdown(
                    """
                    ### Features
                    - 🔍 Smart document search
                    - 🧠 Conversation memory
                    - 🌐 Multi-language support
                    - ⚡ Real-time responses
                    """
                )
        
        # Event handlers
        def respond(message, history):
            """Handle user message and generate bot response"""
            if not message.strip():
                return history, ""
                
            # Add user message to history
            history = history + [[message, None]]
            
            # Generate response using the bot's chat function
            for partial_response in bot.chat_function(message, history):
                history[-1][1] = partial_response
                yield history, ""
            
            return history, ""
        
        def clear_chat():
            """Clear the chat history"""
            return [], ""
        
        def handle_clear_memory():
            """Handle memory clearing"""
            result = bot.clear_memory()
            return result
        
        def handle_status():
            """Handle status request"""
            status = bot.get_system_status()
            return status
        
        # Bind events
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, None, [chatbot, msg])
        clear_memory_btn.click(handle_clear_memory, None, status_output)
        status_btn.click(handle_status, None, status_output)
        
        # Initial status load
        demo.load(handle_status, None, status_output)
    
    return demo


def main():
    """Main function to run the Gradio app"""
    logger.info("Starting Gradio JazzBot application...")
    
    try:
        # Create the interface
        demo = create_gradio_interface()
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default Gradio port
            share=True,            # Set to True if you want a public link
            debug=True,             # Enable debug mode
            show_error=True,        # Show errors in interface
            quiet=False             # Show startup logs
        )
        
    except Exception as e:
        logger.error(f"Failed to start Gradio app: {e}")
        raise


if __name__ == "__main__":
    main()