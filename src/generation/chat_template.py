from typing import List


DEFAULT_SYSTEM_ENTRY = "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."

class ChatTemplate:
    """
    Applies a chat template on a set of entries
    """
    
    def __init__(self, tokenizer, system_entry: str = None, starting_user_entry: str = None):
        """
        Args:
            tokenizer: Tokenizer used to retrieve the chat template from
            system_entry: Initial system entry to add to the generation
            starting_user_entry: Initial user entry to add to the generation
        """
        self.tokenizer = tokenizer
        self.messages = []
        
        if system_entry:
            self.add_system_entry(system_entry)
            
        if starting_user_entry:
            self.add_user_entry(starting_user_entry)
    
    def reset(self):
        """
        Resets all messages in the chat
        """
        self.messages = []
    
    def add_entry(self, role: str, entry: str, add_generation_prompt=True):
        """
        Adds an entry in the messages with given role

        Args:
            role: Role of the entry
            entry: Text to add
            add_generation_prompt: Whether to add the token indicating that it is the model's turn to generate
        """
        self.messages.append({'role': role, 'content': entry})
        return self.apply(self.messages, add_generation_prompt=add_generation_prompt)
    
    def single_user_entry(self, entry: str, system_entry: str = DEFAULT_SYSTEM_ENTRY):
        """
        Returns a chat conversation with a single user entry and the generation prompt added
        """
        return self.apply([
            {"role": "system", "content": system_entry},
            {'role': 'user', 'content': entry}
        ])
    
    def batched_single_user_entry(self, entries: List[str]):
        """
        Returns a chat conversation with a single user entry and the generation prompt added for each entry in the batch
        """
        return list(map(self.single_user_entry, entries))

    def add_user_entry(self, entry: str, add_generation_prompt=True):
        """
        Adds a user entry to the conversation

        Args:
            entry: Text to add
            add_generation_prompt: Whether to add the token indicating that it is the model's turn to generate
        """
        return self.add_entry('user', entry, add_generation_prompt)
        
    def add_assistant_entry(self, entry: str, add_generation_prompt=False):
        """
        Adds an assistant entry to the conversation

        Args:
            entry: Text to add
            add_generation_prompt: Whether to add the token indicating that it is the model's turn to generate
        """
        return self.add_entry('assistant', entry, add_generation_prompt)
        
    def add_system_entry(self, entry: str, add_generation_prompt=True):
        """
        Adds a system entry to the conversation

        Args:
            entry: Text to add
            add_generation_prompt: Whether to add the token indicating that it is the model's turn to generate
        """
        return self.add_entry('system', entry, add_generation_prompt)
    
    def apply(self, messages: list = None, add_generation_prompt=True):
        """
        Transforms a list of messages to the tokenizer's chat template. If `messages` = None, it will use the attribute `self.messages` 
        to generate the conversation

        Args:
            messages: List of messages in the format {'role': role, 'content': content}
            add_generation_prompt: Whether to add the token indicating that it is the model's turn to generate
        """
        if messages is None:
            return self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=add_generation_prompt, return_tensors="pt")
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, return_tensors="pt")
