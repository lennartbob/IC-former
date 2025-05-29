import tiktoken

def count_tokens(text):
    """
    Counts the number of tokens in a given text string using the cl100k_base encoding.

    Args:
        text (str): The input text string.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)