"""
services/prompt.py - Enhanced prompt engineering service for JazzBot
"""

from typing import List, Optional, Dict
from services.analyzer import QueryAnalyzer


class PromptEngine:
    """Enhanced prompt generation service with context optimization"""
    
    def __init__(self, max_context_tokens: int = 2000):
        self.query_analyzer = QueryAnalyzer()
        self.max_context_tokens = max_context_tokens
        
        # Token estimates for prompt sections
        self.base_instructions_tokens = 200
        self.user_query_tokens = 100
        self.buffer_tokens = 300
    
    def get_base_instructions(self) -> str:
        """Get base instructions for JazzBot"""
        return """You are JazzBot, a helpful assistant for Jazz Telecom Pakistan. 

CORE PRINCIPLES:
- Provide accurate information based on the provided knowledge base
- Be conversational but always grammatically correct
- Don't greet every time, only when the user greets you first
- Keep responses concise and to the point (under 150 words unless detailed info requested)
- Always specify the category when mentioning packages (B2C, B2B, etc.)
- Use bullet points for package details when listing features
- Never make up information if it's not in the knowledge base
- If information is incomplete, state what's available and offer to help find more details

CONTEXT UNDERSTANDING:
- ALWAYS check conversation history before responding
- If user says "the first one", "second one", "that package", "it", etc. - look at your previous response to understand what they're referring to
- When you list items (packages, services), remember the order for future reference
- Connect current user question to previous conversation context

RESPONSE FORMAT:
- Direct answer first
- Supporting details with bullet points if needed
- End with helpful follow-up question if relevant"""
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Intelligently truncate context to fit token limits"""
        if self._estimate_tokens(context) <= max_tokens:
            return context
        
        # Split context into chunks and prioritize most relevant
        chunks = context.split('\n\n')
        
        # Calculate target length
        target_chars = max_tokens * 4
        
        # If single chunk is too long, truncate it
        if len(chunks) == 1:
            return context[:target_chars] + "..."
        
        # Otherwise, take most recent chunks that fit
        result_chunks = []
        current_length = 0
        
        for chunk in reversed(chunks):
            if current_length + len(chunk) <= target_chars:
                result_chunks.insert(0, chunk)
                current_length += len(chunk)
            else:
                break
        
        return '\n\n'.join(result_chunks)
    
    def _detect_reference_query(self, user_input: str, memory_context: str = "") -> bool:
        """Detect if user is making a reference to previous conversation"""
        reference_patterns = [
            "first one", "second one", "third one", "last one",
            "that package", "that plan", "that offer", "that service",
            "the first", "the second", "the third", "the last",
            "it", "this one", "that one", "above", "mentioned"
        ]
        user_lower = user_input.lower()
        
        # Check explicit reference patterns
        if any(pattern in user_lower for pattern in reference_patterns):
            return True
        
        # Check semantic context - if user mentions specific package/service names from memory
        if memory_context:
            return self._check_semantic_context(user_input, memory_context)
        
        return False
    
    def _check_semantic_context(self, user_input: str, memory_context: str) -> bool:
        """Check if user input semantically relates to previous conversation"""
        user_lower = user_input.lower()
        memory_lower = memory_context.lower()
        
        # Extract potential package/service names from memory (common Jazz terms)
        jazz_terms = [
            "jazz easycard", "jazz super card", "jazz postpaid", "jazz prepaid",
            "jazz weekly", "jazz monthly", "jazz daily", "jazz hybrid",
            "jazz connect", "jazz business", "jazz enterprise", "jazz fiber",
            "jazz cash", "jazz tv", "jazz 4g", "jazz 5g"
        ]
        
        # Check if user mentions any package/service name that was in previous conversation
        for term in jazz_terms:
            if term in user_lower and term in memory_lower:
                return True
        
        # Check for partial matches (e.g., "super card" when "jazz super card" was mentioned)
        user_words = set(user_lower.split())
        memory_words = set(memory_lower.split())
        
        # If user mentions 2+ words that were in a package name from memory
        significant_words = user_words.intersection(memory_words)
        package_indicators = {"card", "package", "plan", "offer", "bundle", "postpaid", "prepaid"}
        
        if len(significant_words) >= 2 and any(word in significant_words for word in package_indicators):
            return True
        
        # Check for continuation phrases that indicate context dependency
        continuation_phrases = [
            "tell me more", "more details", "what about", "how about",
            "details about", "information about", "price of", "cost of",
            "subscribe to", "activate", "how to get"
        ]
        
        if any(phrase in user_lower for phrase in continuation_phrases):
            # Check if there's overlap in key terms with memory
            if len(significant_words) >= 1 and any(word in significant_words for word in package_indicators):
                return True
        
        return False
    
    def build_context_prompt_main(
        self,
        context: str,
        memory_context: str,
        user_input: str,
        is_followup: bool = False,
        package_name: str = None,
        detected_categories: List[str] = None,
        max_response_tokens: int = 500
    ) -> str:
        """Main method to build optimized context-aware prompt"""
        
        # Calculate available tokens for context
        available_tokens = (
            self.max_context_tokens - 
            self.base_instructions_tokens - 
            self.user_query_tokens - 
            self.buffer_tokens
        )
        
        # Allocate tokens between memory and knowledge context
        memory_tokens = min(available_tokens // 3, 800)  # Max 800 tokens for memory
        knowledge_tokens = available_tokens - memory_tokens
        
        # Truncate contexts
        truncated_memory = self._truncate_context(memory_context, memory_tokens)
        truncated_knowledge = self._truncate_context(context, knowledge_tokens)
        
        # Handle Roman Urdu detection
        if self.query_analyzer.is_roman_urdu(user_input):
            return self.build_roman_urdu_prompt(user_input, truncated_memory)
        
        # Handle reference queries (NEW - highest priority)
        if self._detect_reference_query(user_input, truncated_memory) and truncated_memory:
            return self.build_reference_prompt(
                user_input, truncated_memory, truncated_knowledge
            )
        
        # Handle follow-up with package name
        if is_followup and package_name:
            return self.build_followup_prompt(
                user_input, truncated_memory, truncated_knowledge, package_name
            )
        
        # Handle queries with substantial context
        if truncated_knowledge and len(truncated_knowledge.split()) > 15:
            return self.build_context_prompt(
                user_input, truncated_memory, truncated_knowledge, detected_categories
            )
        
        # Handle queries without context
        else:
            return self.build_no_context_prompt(user_input, truncated_memory)
    
    def build_reference_prompt(self, user_input: str, memory_context: str, knowledge_context: str) -> str:
        """Build prompt specifically for handling references to previous conversation"""
        base_instructions = self.get_base_instructions()
        
        return f"""{base_instructions}

CONVERSATION HISTORY (IMPORTANT - USE THIS TO UNDERSTAND REFERENCES):
{memory_context}

KNOWLEDGE BASE:
{knowledge_context}

REFERENCE RESOLUTION TASK:
The user is referring to something from our previous conversation. 

STEP-BY-STEP PROCESS:
1. Look at the conversation history above
2. Identify what the user is referring to (first one, second one, that package, etc.)
3. Match their reference to the specific item from your previous response
4. Provide detailed information about that specific item

COMMON REFERENCES:
- "first one" = first item you mentioned in your last response
- "second one" = second item you mentioned in your last response  
- "that package/plan" = the package/plan being discussed
- "it" = the main subject of previous conversation

If you listed multiple packages/services before, remember the ORDER.

USER REQUEST: {user_input}

Respond with details about the specific item they're referring to."""
    
    def build_roman_urdu_prompt(self, user_input: str, memory_context: str) -> str:
        """Build optimized prompt for Roman Urdu queries"""
        base_instructions = self.get_base_instructions()

        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

CONTEXT INSTRUCTIONS:
- Check conversation history to understand any references
- If user refers to previous items, identify them from history above

SPECIAL INSTRUCTIONS FOR ROMAN URDU:
- Respond in grammatically correct Roman Urdu ONLY
- Don't use Hindi words or casual expressions
- Use proper structure: "4 hafton ka package", "Jazz Telecom ke packages"
- Keep response under 100 words
- Be direct and accurate
- End with natural Roman Urdu follow-up if relevant

USER REQUEST: {user_input}"""
    
    def build_followup_prompt(self, user_input: str, memory_context: str, context: str, package_name: str) -> str:
        """Build optimized prompt for follow-up queries"""
        base_instructions = self.get_base_instructions()

        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

CONTEXT INSTRUCTIONS:
- Use conversation history to understand any references
- Connect current query to previous discussion

FOLLOWUP TASK: Provide detailed information about "{package_name}"

REQUIRED FORMAT:
• Package Name: [Name]
• Category: [B2C/B2B/etc.]
• Price: [Amount]
• Validity: [Duration]
• Data: [Amount if applicable]
• Calls/SMS: [Details if applicable]
• Subscription: [Code if available]
• Additional Features: [Other benefits]

If any detail is missing, write "Not specified"

USER REQUEST: {user_input}"""
    
    def build_context_prompt(self, user_input: str, memory_context: str, context: str, detected_categories: List[str] = None) -> str:
        """Build optimized prompt with knowledge base context"""
        base_instructions = self.get_base_instructions()
        
        category_hint = ""
        if detected_categories:
            category_hint = f"Focus on: {', '.join(detected_categories[:2])}"
        
        return f"""{base_instructions}

CONVERSATION HISTORY (CHECK FOR REFERENCES):
{memory_context}

KNOWLEDGE BASE:
{context}

{category_hint}

CONTEXT INSTRUCTIONS:
- FIRST check conversation history for any references (first one, that package, etc.)
- If user is referencing something from previous conversation, identify it
- Use both conversation context and knowledge base to provide complete answer

INSTRUCTIONS:
- Answer based on the knowledge provided above
- Keep response under 120 words unless detailed info explicitly requested
- Specify category when mentioning packages
- If detailed information available, end with "Would you like more details?"
- Be specific about what services are available

USER REQUEST: {user_input}"""

    def build_no_context_prompt(self, user_input: str, memory_context: str) -> str:
        """Build prompt when no relevant context is available"""
        base_instructions = self.get_base_instructions()

        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

CONTEXT INSTRUCTIONS:
- Even without specific knowledge, check conversation history for references
- If user refers to something discussed before, acknowledge it

NO SPECIFIC KNOWLEDGE AVAILABLE

GUIDANCE RESPONSES:
- For greetings: "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
- For general queries: "I'd be happy to help! Could you specify what Jazz service you need? For example: B2C packages, B2B solutions, data offers, or call plans?"
- For references without context: "I see you're referring to something we discussed, but I need more specific information to help you better."
- Keep under 50 words
- Guide user to be more specific
- Don't repeat greetings

USER REQUEST: {user_input}"""
    
    def build_greeting_response(self, user_input: str) -> str:
        """Build appropriate greeting response"""
        user_lower = user_input.lower()
        
        if any(greeting in user_lower for greeting in ["salam", "assalam", "aoa"]):
            return "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
        else:
            return "Hello! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
    
    def build_error_prompt(self, error_type: str = "general") -> str:
        """Build error handling prompts"""
        error_responses = {
            "no_llm": "AI service is temporarily unavailable. Please try again in a moment.",
            "no_search": "Search functionality is currently unavailable. I can still help with general Jazz Telecom questions.",
            "timeout": "Request timed out. Please try asking your question again.",
            "general": "I encountered an error processing your request. Please try again."
        }
        
        return error_responses.get(error_type, error_responses["general"])
    
    def optimize_prompt_length(self, prompt: str, max_tokens: int) -> str:
        """Optimize prompt length while preserving important information"""
        if self._estimate_tokens(prompt) <= max_tokens:
            return prompt
        
        # Split prompt into sections
        sections = prompt.split('\n\n')
        
        # Identify critical sections (base instructions, user request, conversation history)
        critical_sections = []
        optional_sections = []
        
        for section in sections:
            if any(keyword in section for keyword in [
                "CORE PRINCIPLES:", "USER REQUEST:", "CONVERSATION HISTORY:", 
                "CONTEXT INSTRUCTIONS:", "REFERENCE RESOLUTION TASK:"
            ]):
                critical_sections.append(section)
            else:
                optional_sections.append(section)
        
        # Build optimized prompt
        result = '\n\n'.join(critical_sections)
        target_chars = max_tokens * 4
        
        # Add optional sections if space allows
        for section in optional_sections:
            if len(result) + len(section) <= target_chars:
                result += '\n\n' + section
            else:
                # Add truncated version
                remaining_space = target_chars - len(result) - 20
                if remaining_space > 100:
                    result += '\n\n' + section[:remaining_space] + "..."
                break
        
        return result