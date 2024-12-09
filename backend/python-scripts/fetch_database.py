import json
from rapidfuzz import fuzz

def book_exists(database, filters, title_weight=0.5):  # title_weight = 70%, author_weight = 30%
    """
    Search for a book in the database using fuzzy matching.
    
    Args:
        database (list): List of book dictionaries
        filters (dict): Dictionary containing search filters (title, author, isbn10, isbn13)
        title_weight (float): Weight given to title match (0-1)
    
    Returns:
        dict: Dictionary containing matched book and match scores, or None if no match found
    """
    best_match = None
    highest_score = 0

    for book in database:
        # Check by ISBN first (exact match)
        if filters.get("isbn10") or filters.get("isbn13"):
            if (
                (filters.get("isbn10") and filters["isbn10"] in book.get("isbn10", [])) or
                (filters.get("isbn13") and filters["isbn13"] in book.get("isbn13", []))
            ):
                return {
                    "book": book,
                    "score": 100,
                    "title_similarity": 100,
                    "author_similarity": 100
                }
        
        # Calculate similarity scores
        title_ratio = 0
        author_ratio = 0
        
        if filters.get("title"):
            title_ratio = fuzz.ratio(
                book.get("title", "").lower(),
                filters["title"].lower()
            )
        
        if filters.get("author"):
            author_ratio = fuzz.ratio(
                book.get("author", "").lower(),
                filters["author"].lower()
            )
        
        # Calculate weighted combined score
        if filters.get("title") and filters.get("author"):
            author_weight = 1 - title_weight
            combined_score = (title_ratio * title_weight) + (author_ratio * author_weight)
        elif filters.get("title"):
            combined_score = title_ratio
        else:
            combined_score = 0
            
        # # Prioritize exact title matches
        # if book.get("title", "").lower() == filters.get("title", "").lower():
        #     combined_score = author_ratio + 100  # Adding 100 ensures exact matches rank higher
        
        # Update best match if this is the highest score so far
        if combined_score > highest_score:
            highest_score = combined_score
            best_match = {
                "book": book,
                "score": highest_score,
                "title_similarity": title_ratio,
                "author_similarity": author_ratio
            }

    # Return match only if score is above threshold
    if highest_score > 70:
        return best_match
    else:
        return None

if __name__ == "__main__":
    with open("../database/bookMeta.jsonl.json", "r", encoding="utf-8") as file:
        database = [json.loads(line) for line in file]
    print(book_exists(database, {"title": "wonder", "author": "selznick"}))
