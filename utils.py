"""
utils.py - Utility classes and functions for JazzBot.

This file contains classes and functions for conversation memory management, document processing,
category detection, query analysis, prompt generation, and other helpers.
"""

import os
import re
import json
import hashlib
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jazzbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-local storage for request context
thread_local = threading.local()

class ConversationMemory:
    """Thread-safe conversation memory management"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
    
    def get_memory(self) -> List[Dict]:
        """Get conversation history - thread-safe"""
        try:
            # First try to get from thread local (during streaming)
            if hasattr(thread_local, 'chat_memory'):
                return thread_local.chat_memory
            
            # Otherwise get from session
            from flask import session
            if 'chat_memory' not in session:
                session['chat_memory'] = []
            return session['chat_memory']
        except:
            # Fallback to empty memory if session unavailable
            return getattr(thread_local, 'chat_memory', [])
    
    def add_exchange(self, question: str, answer: str, category: Optional[str] = None):
        """Add a Q&A pair to memory - thread-safe"""
        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Update session memory if available
            from flask import session
            memory = session.get('chat_memory', [])
            memory.append(entry)
            if len(memory) > self.max_size:
                memory = memory[-self.max_size:]
            session['chat_memory'] = memory
            
            # Also update thread local
            thread_local.chat_memory = memory
            
        except:
            # If session unavailable, use thread local only
            if not hasattr(thread_local, 'chat_memory'):
                thread_local.chat_memory = []
            thread_local.chat_memory.append(entry)
            if len(thread_local.chat_memory) > self.max_size:
                thread_local.chat_memory = thread_local.chat_memory[-self.max_size:]
        
        logger.info(f"Added to memory: Q='{question[:50]}...', Category={category}")
    
    def get_context_string(self, include_last_n: int = 3) -> str:
        """Generate formatted context string from recent memory"""
        memory = self.get_memory()
        if not memory:
            return ""
        
        recent_memory = memory[-include_last_n:] if include_last_n > 0 else memory
        context_parts = []
        
        for i, entry in enumerate(recent_memory, 1):
            context_parts.append(f"Previous Q{i}: {entry['question']}")
            context_parts.append(f"Previous A{i}: {entry['answer']}")
        
        return "\n".join(context_parts)
    
    def get_last_response(self) -> str:
        """Get the last response"""
        memory = self.get_memory()
        return memory[-1]["answer"] if memory else ""
    
    def clear_memory(self):
        """Clear conversation memory"""
        from flask import session
        session['chat_memory'] = []
        if hasattr(thread_local, 'chat_memory'):
            delattr(thread_local, 'chat_memory')

class DocumentProcessor:
    """Enhanced document processing and retrieval"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\.\,\:\;\(\)\[\]\/\&\%\$\#\@\!]', ' ', text)
        return text.strip()
    
    @staticmethod
    def load_documents_with_categories(data_dir: str) -> Tuple[List, List[str]]:
        """Load documents with enhanced metadata"""
        docs = []
        file_paths = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.warning(f"Created {data_dir} directory - no documents found")
            return docs, file_paths
        
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            for file in os.listdir(category_path):
                if not file.endswith(".txt"):
                    continue
                    
                file_path = os.path.join(category_path, file)
                file_paths.append(file_path)
                
                # Determine file type with better logic
                file_type = "unknown"
                file_lower = file.lower()
                if "offer" in file_lower:
                    file_type = "offers"
                elif "package" in file_lower:
                    file_type = "packages"
                elif "data" in file_lower:
                    file_type = "data_offers"
                elif "bundle" in file_lower:
                    file_type = "bundles"
                
                try:
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                    
                    for doc in documents:
                        doc.page_content = DocumentProcessor.clean_text(doc.page_content)
                        doc.metadata.update({
                            "category": category,
                            "file_type": file_type,
                            "filename": file,
                            "source_path": file_path
                        })
                    
                    docs.extend(documents)
                    logger.info(f"Loaded {len(documents)} documents from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(docs)}")
        return docs, file_paths

class CategoryDetector:
    """Enhanced category detection"""
    
    CATEGORY_KEYWORDS = {
        "B2C": ["b2c", "consumer", "individual", "personal"],
        "B2B": ["b2b", "business", "corporate", "enterprise"],
        "prepaid": ["prepaid", "pre-paid", "recharge", "top-up"],
        "postpaid": ["postpaid", "post-paid", "monthly", "contract"],
        "data": ["data", "internet", "mb", "gb", "4g", "5g"],
        "voice": ["call", "calls", "minutes", "voice", "talk"],
        "sms": ["sms", "text", "message", "messaging"],
        "facebook": ["facebook", "fb"],
        "youtube": ["youtube", "yt"],
        "whatsapp": ["whatsapp", "wa"]
    }
    
    @classmethod
    def detect_categories(cls, query: str) -> List[str]:
        """Detect multiple categories from query"""
        query_lower = query.lower()
        detected = []
        
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(category)
        
        return detected
    
    @classmethod
    def detect_primary_category(cls, query: str) -> Optional[str]:
        """Detect primary category"""
        categories = cls.detect_categories(query)
        return categories[0] if categories else None

class QueryAnalyzer:
    """Enhanced query analysis"""
    
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|salam|assalam|good\s+(morning|afternoon|evening))$',
        r'^(how\s+are\s+you|what\'s\s+up)$'
    ]
    
    FOLLOWUP_PATTERNS = [
        r'^(yes|yeah|yep|sure|ok|okay)$',
        r'^(show\s+me\s+)?(more\s+)?(details?|info)',
        r'^(tell\s+me\s+more|give\s+me\s+more)',
        r'^(what\s+about|how\s+about)',
        r'^(can\s+you\s+explain|explain\s+more)'
    ]
    
    PACKAGE_EXTRACTION_PATTERNS = [
        r'(?:we have|offers?|try)\s+(?:a\s+|the\s+)?([^.]+?)(?:\s+(?:offer|bundle|package|plan))',
        r'(facebook|youtube|whatsapp|instagram)\s+(?:premium\s+|daily\s+|weekly\s+|monthly\s+)?(?:offer|bundle|package)',
        r'(?:jazz\s+)?(super\s+card|smart\s+bundle)'
    ]
    
    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Check if query is a greeting"""
        query_clean = query.strip().lower()
        return any(re.match(pattern, query_clean, re.IGNORECASE) for pattern in cls.GREETING_PATTERNS)
    
    @classmethod
    def is_followup(cls, query: str, last_response: str = "") -> bool:
        """Check if query is a follow-up question"""
        query_clean = query.strip().lower()
        is_followup_pattern = any(re.match(pattern, query_clean, re.IGNORECASE) for pattern in cls.FOLLOWUP_PATTERNS)
        has_details_offer = any(phrase in last_response.lower() for phrase in [
            "would you like more details", "want more details", "details?", "more info"
        ])
        return is_followup_pattern and has_details_offer
    
    @classmethod
    def extract_package_name(cls, text: str) -> Optional[str]:
        """Extract package name from text"""
        for pattern in cls.PACKAGE_EXTRACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

class PromptEngine:
    """Enhanced prompt generation"""
    
    @staticmethod
    def build_context_prompt(context: str, memory_context: str, user_input: str, 
                           is_followup: bool = False, package_name: str = None,
                           detected_categories: List[str] = None) -> str:
        """Build enhanced context-aware prompt"""
        
        base_instructions = """You are JazzBot, a helpful assistant for Jazz Telecom Pakistan. 

CORE PRINCIPLES:
- Provide accurate information based on the provided knowledge
- Be conversational and friendly
- Keep responses concise and to the point
- Always specify the category when mentioning packages (B2C, B2B, etc.)
- Use bullet points for detailed information
- Don't make up information if not in the knowledge base"""

        if is_followup and package_name:
            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

TASK: The user has requested all available details about "{package_name}". Provide a comprehensive response including every piece of information available in the knowledge base, formatted in bullet points.

REQUIRED FORMAT (include all fields, even if not applicable):
- Package Name: [Name]
- Category: [B2C/B2B/etc.]
- Offer ID: [ID if available]
- Offer Category: [Prepaid/Postpaid/etc.]
- Price: [Amount]
- Validity: [Duration]
- Consumable Data/Internet: [Amount if applicable]
- On-Network Calls: [Details if applicable]
- Off-Network Calls: [Details if applicable]
- On-Network SMS: [Details if applicable]
- Off-Network SMS: [Details if applicable]
- International SMS: [Details if applicable]
- Subscription Code: [Code if available]
- Unsubscription Code: [Code if available]
- Time Window (Usage Hours): [Hours if specified]
- Is Recursive: [Yes/No if specified]
- Is Prorated: [Yes/No if specified]
- Description: [Description if available]
- Incentive: [Incentive if available]
- Base Rate: [Rate if applicable]
- Notify Domain: [Domain if specified]
- Data Type: [Type if specified]
- Other Details: [Any additional info]

If any information is not available, write "Not specified in available data"

USER REQUEST: {user_input}"""

        elif context and len(context.split()) > 15:
            category_info = f"Detected categories: {', '.join(detected_categories)}" if detected_categories else ""
            
            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

{category_info}

INSTRUCTIONS:
- Answer based on the knowledge provided above
- Keep response under 100 words
- Specify category when mentioning packages (B2C, B2B, etc.)
- If you have detailed info available, end with "Would you like more details?"
- Be specific about what's available
- For user input like AOA or aoa or Assalam o alaikum: "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"

USER REQUEST: {user_input}"""

        else:
            return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

SITUATION: No specific knowledge available for this query.

INSTRUCTIONS:
- For greetings: "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
- For specific queries: "I'd be happy to help! Could you please specify what type of Jazz service you're looking for? For example: B2C packages, B2B solutions, data offers, voice plans, or SMS bundles?"
- Keep response under 50 words
- Be helpful and guide the user to provide more specific information
- For user input like AOA or aoa or Assalam o alaikum: "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"

USER REQUEST: {user_input}"""

def compute_file_hash(file_paths: List[str]) -> str:
    """Compute hash of input files"""
    hasher = hashlib.md5()
    for file_path in sorted(file_paths):
        try:
            with open(file_path, "rb") as f:
                hasher.update(f.read())
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
    return hasher.hexdigest()

def save_response(question: str, response: str, category: str = None, responses_dir: str = "responses"):
    """Save response to file"""
    try:
        os.makedirs(responses_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = os.path.join(responses_dir, f"response_{timestamp}.txt")
        
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Category: {category or 'Unknown'}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
    except Exception as e:
        logger.error(f"Error saving response: {e}")