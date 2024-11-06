import re
from string import punctuation
import nltk
from nltk.corpus import words, wordnet
from nltk.tokenize import word_tokenize

class TextValidator:
    def __init__(self):
        # Download required NLTK data (run once)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('words')
        
        # Common internet slang/expressions that might not be in NLTK
        self.internet_slang = {
            'lol', 'omg', 'wtf', 'lmao', 'rofl', 'idk', 'tbh',
            'haha', 'hehe', 'hmm', 'ohh', 'ahh'
        }

    def has_excessive_repetition(self, word: str) -> bool:
        """
        Check if word has excessive character repetition
        Returns True for cases like 'aaaaa', 'aaaaabbbbb'
        """
        # Check for single character repetition (e.g., 'aaaaa')
        if len(set(word)) == 1 and len(word) > 2:
            return True
            
        # Check for patterns of repetition
        for char in set(word):
            if word.count(char) > 3: 
                consecutive_count = max(len(match[0]) for match in re.finditer(f'{char}+', word))
                if consecutive_count > 2:  
                    return True
        return False

    def is_meaningful_word(self, word: str) -> bool:
        """
        Check if a word is meaningful using NLTK WordNet and repetition rules
        """
        # Clean the word
        word = ''.join(c for c in word.lower() if c not in punctuation)
        if not word:
            return False

        # Check length
        if len(word) < 2:
            return False

        # Check for excessive repetition first
        if self.has_excessive_repetition(word):
            return False

        # Check internet slang
        if word in self.internet_slang:
            return True

        # Remove doubled letters and check if base word exists
        base_word = re.sub(r'(.)\1+', r'\1', word)
        
        # If word exists in WordNet or is a common word
        if wordnet.synsets(word) or word in words.words():
            return True
            
        # If base word exists in WordNet (for cases like 'sooo' -> 'so')
        if wordnet.synsets(base_word):
            # Only allow if it's a reasonable extension
            return len(word) <= len(base_word) + 2  # More strict: max 2 extra chars
            
        return False

    def validate_review_text(self, text: str) -> tuple[bool, str]:
        """
        Validate the review text using NLTK and repetition checks
        """
        if not text or text.isspace():
            return False, "Review text cannot be empty"

        if len(text) > 512:
            return False, "Review text is too long. Maximum length is 512 characters"

        # Remove extra spaces and tokenize
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        
        # Filter out punctuation tokens
        words = [word for word in tokens if any(c.isalnum() for c in word)]
        
        if not words:
            return False, "Review contains no meaningful words"

        # Check each word
        for word in words:
            if self.has_excessive_repetition(word):
                return False, f"Invalid word with excessive repetition: {word}"
            if not self.is_meaningful_word(word):
                return False, f"Invalid or meaningless word: {word}"

        return True, ""
