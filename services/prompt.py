"""
services/prompt.py - Prompt engineering service for JazzBot

This module handles all prompt generation and formatting for different types of queries.
"""

from typing import List, Optional
from services.analyzer import QueryAnalyzer


class PromptEngine:
    """Enhanced prompt generation service"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
    
    def get_base_instructions(self) -> str:
        """Get base instructions for JazzBot"""
        return """You are JazzBot, a helpful assistant for Jazz Telecom Pakistan. 

CORE PRINCIPLES:
- Provide accurate information based on the provided knowledge.
- Be conversational but always grammatically correct.
- Don't greet everytime, only if the user greets you.
- Keep responses concise and to the point.
- Always specify the category when mentioning packages (B2C, B2B, etc.).
- Use bullet points for details when listing features.
- Never make up information if it's not in the knowledge base."""
    
    def build_roman_urdu_prompt(self, user_input: str, memory_context: str) -> str:
        """Build prompt for Roman Urdu queries"""
        base_instructions = self.get_base_instructions()
        
        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

SITUATION: Roman Urdu query detected.

INSTRUCTIONS:
- Respond in **grammatically correct Roman Urdu**.
- Don't greet everytime, only if the user greets you.
- Don't use any Hindi words or phrases.
- Provide concise and accurate information based on the knowledge base.
- Start your response with right answer.
- Do not use casual or incorrect phrases like "bohat hi aam hain" or "humnein 4 hafte ki package".
- Use correct structure: e.g., "4 hafton ka package", "Jazz Telecom ke packages", etc.
- Do **not** include any English phrases like "Follow-up question".
- End with a natural Roman Urdu question if relevant, like:  
  ➤ "Kya aap kisi aur package ke baray mein maloomat lena chahenge?"
- Keep the tone formal — no slang, no informal speech.


USER REQUEST: {user_input}"""
    
    def build_followup_prompt(self, user_input: str, memory_context: str, context: str, package_name: str) -> str:
        """Build prompt for follow-up queries"""
        base_instructions = self.get_base_instructions()
        
        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

TASK: You are being asked for more details about "{package_name}". Provide comprehensive information in bullet point format.

REQUIRED FORMAT:
- Package Name: [Name]
- Category: [B2C/B2B/etc.]
- Price: [Amount]
- Validity: [Duration]
- Data/Internet: [Amount if applicable]
- Calls: [Details if applicable]
- SMS: [Details if applicable]
- Subscription Code: [Code if available]
- Other Details: [Any additional info]

If any information is not available, write "Not specified in available data"

USER REQUEST: {user_input}"""
    
    def build_context_prompt(self, user_input: str, memory_context: str, context: str, detected_categories: List[str] = None) -> str:
        """Build prompt with context from knowledge base"""
        base_instructions = self.get_base_instructions()
        category_info = (
            f"Detected categories: {', '.join(detected_categories)}"
            if detected_categories
            else ""
        )
        
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

USER REQUEST: {user_input}"""
    
    def build_no_context_prompt(self, user_input: str, memory_context: str) -> str:
        """Build prompt when no context is available"""
        base_instructions = self.get_base_instructions()
        
        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

SITUATION: No specific knowledge available for this query.

INSTRUCTIONS:
- For greetings: "Hi! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
- For specific queries: "I'd be happy to help! Could you please specify what type of Jazz service you're looking for? For example: B2C packages, B2B solutions, data offers, voice plans, or SMS bundles?"
- Keep response under 50 words
- Don't greet everytime, only if the user greets you.
- Be helpful and guide the user to provide more specific information

USER REQUEST: {user_input}"""
    
    def build_context_prompt_main(
        self,
        context: str,
        memory_context: str,
        user_input: str,
        is_followup: bool = False,
        package_name: str = None,
        detected_categories: List[str] = None,
    ) -> str:
        """Main method to build context-aware prompt"""
        
        # Handle Roman Urdu detection
        if self.query_analyzer.is_roman_urdu(user_input):
            return self.build_roman_urdu_prompt(user_input, memory_context)
        
        # Handle follow-up with package name
        if is_followup and package_name:
            return self.build_followup_prompt(user_input, memory_context, context, package_name)
        
        # Handle queries with context
        elif context and len(context.split()) > 15:
            return self.build_context_prompt(user_input, memory_context, context, detected_categories)
        
        # Handle queries without context
        else:
            return self.build_no_context_prompt(user_input, memory_context)
    
    def build_greeting_response(self, user_input: str) -> str:
        """Build appropriate greeting response"""
        if any(
            word in user_input.lower()
            for word in [
                "salam",
                "assalam",
                "aoa",
                "assalam o alaikum",
                "slam",
                "assalam",
            ]
        ):
            return "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
        else:
            return "Hello! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"