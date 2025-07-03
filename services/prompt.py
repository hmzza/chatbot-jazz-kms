"""
services/prompt.py - Prompt engineering service for JazzBot

This module handles all prompt generation and formatting for different types of queries.
"""

from typing import List, Optional
from services.analyzer import QueryAnalyzer


class PromptEngine:
    """Prompt generation service"""

    def __init__(self):
        self.query_analyzer = QueryAnalyzer()

    def get_base_instructions(self) -> str:
        """Get base instructions for JazzBot"""
        return """You are JazzBot, a helpful assistant for Jazz Telecom Pakistan. 

CORE PRINCIPLES:
- Provide accurate information based ONLY on the provided knowledge base.
- Do NOT generate information not present in the knowledge base.
- Be conversational but always grammatically correct.
- Don't greet unless the user greets you.
- Keep responses concise, under 100 words.
- Always specify the category when mentioning packages (B2C, B2B, etc.).
- Use bullet points for package details.
- If information is missing, say: "Sorry, I don't have details for that. Can you clarify or ask about another package?" """

    def build_followup_prompt(
        self, user_input: str, memory_context: str, context: str, package_name: str
    ) -> str:
        """Build prompt for follow-up queries"""
        base_instructions = self.get_base_instructions()

        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

KNOWLEDGE BASE:
{context}

TASK: Provide details about "{package_name}" based ONLY on the knowledge base.

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

If information is missing, say: "Not specified in available data."

USER REQUEST: {user_input}"""

    def build_context_prompt(
        self,
        user_input: str,
        memory_context: str,
        context: str,
        detected_categories: List[str] = None,
    ) -> str:
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
- Answer based ONLY on the knowledge provided above.
- Keep response under 100 words.
- Specify category when mentioning packages (B2C, B2B, etc.).
- If no relevant information, say: "Sorry, I don't have details for that. Can you clarify?"
- End with: "Would you like more details?"

USER REQUEST: {user_input}"""

    def build_no_context_prompt(self, user_input: str, memory_context: str) -> str:
        """Build prompt when no context is available"""
        base_instructions = self.get_base_instructions()

        return f"""{base_instructions}

CONVERSATION HISTORY:
{memory_context}

SITUATION: No specific knowledge available for this query.

INSTRUCTIONS:
- For greetings: "Hello! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
- For specific queries: "I'd be happy to help! Could you please specify what type of Jazz service you're looking for? For example: B2C packages, B2B solutions, data offers, voice plans, or SMS bundles?"
- Keep response under 50 words.

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
        if is_followup and package_name:
            return self.build_followup_prompt(
                user_input, memory_context, context, package_name
            )
        elif context and len(context.split()) > 15:
            return self.build_context_prompt(
                user_input, memory_context, context, detected_categories
            )
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
                "aoa",
                "salam o alaikum",
                "salam walaikum",
                "assalam",
            ]
        ):
            return "Walaikum Salam! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
        else:
            return "Hello! I'm JazzBot, your Jazz Telecom assistant. How can I help you today?"
