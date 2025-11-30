"""Conversation memory with SQL persistence"""
import uuid
from typing import Optional
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemoryManager:
    """
    Manages conversation memory with SQL persistence
    
    Uses LangChain's SQLChatMessageHistory for persistence.
    """
    
    def __init__(self, db_connection_string: str = "sqlite:///data/conversations.db"):
        """
        Initialize conversation memory manager
        
        Args:
            db_connection_string: SQLAlchemy connection string
        """
        self.db_connection_string = db_connection_string
        self.sessions = {}
        
        logger.info(f'ConversationMemoryManager initialized with {db_connection_string}')
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            session_id: Optional session ID (generates UUID if not provided)
        
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            from langchain.memory import ConversationBufferMemory
            from langchain_community.chat_message_histories import SQLChatMessageHistory
            
            # Create SQL chat history
            chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=self.db_connection_string
            )
            
            # Create memory with SQL persistence
            memory = ConversationBufferMemory(
                chat_memory=chat_history,
                return_messages=True,
                memory_key="chat_history"
            )
            
            self.sessions[session_id] = memory
            logger.info(f'Created conversation session: {session_id}')
            
            return session_id
            
        except ImportError:
            logger.warning('langchain-community not installed, using simple memory')
            # Fallback to simple dict-based memory
            self.sessions[session_id] = {
                'messages': [],
                'session_id': session_id
            }
            return session_id
    
    def get_session(self, session_id: str):
        """
        Get conversation memory for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Memory object or None
        """
        if session_id not in self.sessions:
            logger.warning(f'Session not found: {session_id}, creating new session')
            return self.create_session(session_id)
        
        return self.sessions[session_id]
    
    def save_context(self, session_id: str, input_text: str, output_text: str):
        """
        Save input-output pair to conversation memory
        
        Args:
            session_id: Session ID
            input_text: User input
            output_text: Agent output
        """
        memory = self.get_session(session_id)
        
        if hasattr(memory, 'save_context'):
            # LangChain memory
            memory.save_context(
                {"input": input_text},
                {"output": output_text}
            )
            logger.debug(f'Saved context to session {session_id}')
        else:
            # Fallback dict-based memory
            memory['messages'].append({
                'input': input_text,
                'output': output_text
            })
    
    def load_context(self, session_id: str) -> dict:
        """
        Load conversation history for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Dictionary with conversation history
        """
        memory = self.get_session(session_id)
        
        if hasattr(memory, 'load_memory_variables'):
            # LangChain memory
            return memory.load_memory_variables({})
        else:
            # Fallback dict-based memory
            return {'messages': memory.get('messages', [])}
    
    def clear_session(self, session_id: str):
        """
        Clear conversation memory for a session
        
        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            memory = self.sessions[session_id]
            
            if hasattr(memory, 'clear'):
                memory.clear()
            
            del self.sessions[session_id]
            logger.info(f'Cleared session: {session_id}')
    
    def list_sessions(self) -> list:
        """
        List all active session IDs
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())


# Global memory manager instance
_memory_manager = None


def get_memory_manager(db_connection_string: Optional[str] = None) -> ConversationMemoryManager:
    """
    Get or create global memory manager
    
    Args:
        db_connection_string: SQLAlchemy connection string
    
    Returns:
        ConversationMemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        import os
        db_string = db_connection_string or os.getenv('CONVERSATION_DB', 'sqlite:///data/conversations.db')
        _memory_manager = ConversationMemoryManager(db_string)
    return _memory_manager
