import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from detect import BookDetector
from fetch_book_info import fetch_book_data, fetch_book_data_by_isbn
from fetch_database import book_exists

def load_database(database_path: str = "../database/bookMeta.jsonl.json") -> List[Dict]:
    """
    Load and initialize the book database.
    
    Args:
        database_path (str): Path to the database file
        
    Returns:
        list: List of book dictionaries
    """
    database = []
    try:
        with open(database_path, "r", encoding='utf-8') as file:
            for line in file:
                try:
                    book = json.loads(line.strip())
                    database.append(book)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
        print(f"Loaded {len(database)} books from database")
        return database
    except Exception as e:
        print(f"Error loading database: {e}")
        return []

async def process_image(image_path: str, database: List[Dict]):
    try:
        detector = BookDetector(
            "../models/yolo_weights/best.pt",
            "AIzaSyDHTHu1hqbZkNE1CvU_tF2gnUtGC-GHOKo",
            image_path
        )
        
        # Check if output.json already exists for this image
        output_path = f"{detector.output_dir}/output.json"
        if Path(output_path).exists():
            with open(output_path, 'r') as file:
                existing_results = json.load(file)
                print("BOOK_DETECTION_RESULT:" + json.dumps(existing_results))
                return 0
        
        # If no existing results, proceed with detection and processing
        detected_books = await detector.process_image_async(image_path)
        
        enriched_books = []
        for book in detected_books:
            # Start with the detected data as base
            # print(book)
            enriched_book = {
                "title": book.get('title'),
                "author": book.get('author'),
                "detectedText": book.get('rawText'),
                "isValid": book.get('isValid', False),
                "source": "detection_only",  # Default source
                "original_title": book.get('title'),  # Store original detection
                "original_author": book.get('author')  # Store original detection
            }
            
            if not book.get('isValid', False):
                enriched_books.append(enriched_book)
                continue
            
            # Try database first
            db_result = book_exists(database, {
                "title": book.get('title', ''),
                "author": book.get('author', '')
            })
            
            if db_result and db_result.get('book'):
                # Use database result and format it
                formatted_book = format_book_data(db_result['book'])
                formatted_book.update({
                    "matchScore": db_result['score'],
                    "source": 'database',  # Set source as database
                })
                enriched_book.update(formatted_book)
            else:
                # Try web APIs
                try:
                    web_data = fetch_book_data(
                        book.get('title', ''),
                        book.get('author', '')
                    )
                    
                    if web_data:
                        web_book = json.loads(web_data)
                        formatted_web_book = format_book_data(web_book)
                        formatted_web_book.update(enriched_book)
                        formatted_web_book.update({
                            "isValid": True,
                            "source": 'web'
                        })
                        
                        # If we got an ISBN, try to get more details
                        isbn = (formatted_web_book.get('isbn_13', []) + formatted_web_book.get('isbn_10', []))[0] if (formatted_web_book.get('isbn_13') or formatted_web_book.get('isbn_10')) else None
                        if isbn:
                            isbn_data = fetch_book_data_by_isbn(isbn)
                            if isbn_data:
                                isbn_book = json.loads(isbn_data)
                                formatted_isbn_book = format_book_data(isbn_book)
                                # data soourse is database + web
                                formatted_web_book.update(formatted_isbn_book)
                                formatted_web_book.update({
                                    "source": 'database_and_web'
                                })
                                # Merge data, preferring ISBN data for missing fields
                                for key, value in formatted_isbn_book.items():
                                    if not formatted_web_book.get(key) and value:
                                        formatted_web_book[key] = value
                        
                        enriched_book = formatted_web_book
                    else:
                        formatted_book = format_book_data({
                            "title": book.get('title'),
                            "author": book.get('author'),
                            
                        })
                        enriched_book.update(formatted_book)
                        
                except Exception as e:
                    print(f"Error fetching web data: {e}")
                    enriched_book = format_book_data({
                        "title": book.get('title'),
                        "author": book.get('author'),
                        "detectedText": book.get('rawText'),
                        "isValid": True,
                        "matchScore": db_result['score'],
                        "source": 'database',  # Set source as database
                        # "original_title": book.get('title'),
                        # "original_author": book.get('author')
                    })
                    enriched_book.update(enriched_book)
            
            # Remove any None values from lists
            for key, value in enriched_book.items():
                if isinstance(value, list):
                    enriched_book[key] = [v for v in value if v is not None]
            
            enriched_books.append(enriched_book)
        
        # Dump the result to the output directory
        with open(f"{detector.output_dir}/output.json", "w") as file:
            json.dump(enriched_books, file)

        print("BOOK_DETECTION_RESULT:" + json.dumps(enriched_books))
        return 0
        
    except Exception as e:
        print("here")
        import traceback
        print(traceback.format_exc())
        print("BOOK_DETECTION_RESULT:" + json.dumps({
            "error": str(e.with_traceback(sys.exc_info()[2])),
            "success": False
        }))
        return 1

def format_book_data(db_book):
    """Format book data from database entry"""
    
    # Basic fields - direct access
    formatted_data = {
        "title": db_book.get('title'),
        "subtitle": db_book.get('subtitle'),
        "creators": {
            "authors": [db_book.get('author')] if db_book.get('author') else [],
            "illustrators": [db_book.get('illustrator')] if db_book.get('illustrator') else [],
            "editors": [db_book.get('editor')] if db_book.get('editor') else [],
        },
        
        # Series info
        "series": {
            "name": db_book.get('series'),
            "position": db_book.get('seriesBookNumber')
        },
        
        # Classification
        "genres": db_book.get('genre', []),
        "form": db_book.get('narrativeForm', []),
        "book_type": ("Fiction" if db_book.get('isFiction') else 
                    "Non-Fiction" if db_book.get('isNonFiction') else 
                    "Blended" if db_book.get('isBlended') else "N/A"),
        
        # Edition info
        "isbn_10": db_book.get('isbn10', []),
        "isbn_13": db_book.get('isbn13', []),
        
        # Single image URL
        "cover_image": None,
        
        # Additional info
        "additional_info": {
            "subjects": db_book.get('subjects', []),
            "publisher": [{"name": pub} for pub in db_book.get('publisher', [])] if isinstance(db_book.get('publisher'), list) else [],
            "publish_places": [{"name": place} for place in db_book.get('publish_places', [])] if isinstance(db_book.get('publish_places'), list) else [],
            "languages": db_book.get('languages', []),
            "covers": db_book.get('covers', {})
        }
    }

    # Get edition-specific info
    editions = db_book.get('editions', [])
    if editions:
        first_edition = editions[0]
        formatted_data.update({
            "format": first_edition.get('format'),
            "page_count": first_edition.get('pageCount'),
            "copyright_date": first_edition.get('copyrightDate'),
        })
        
        # Try to get cover image from edition
        formatted_data["cover_image"] = (
            first_edition.get('image') or 
            (first_edition.get('images', []) and first_edition['images'][0]) or 
            None
        )
    
    # If no cover image was set from editions, try getting it from various sources
    if not formatted_data["cover_image"]:
        # Try root level image first
        formatted_data["cover_image"] = (
            db_book.get('image') or 
            (db_book.get('images', []) and db_book['images'][0]) or 
            None
        )
        
        # If still no image, try additional_info covers
        if not formatted_data["cover_image"]:
            covers = formatted_data["additional_info"]["covers"]
            if covers:
                # Prefer medium size, fall back to large, then small
                formatted_data["cover_image"] = (
                    covers.get('medium') or 
                    covers.get('large') or 
                    covers.get('small')
                )

    return formatted_data

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        return 1
    
    # Load database once at startup
    database = load_database()
    
    image_path = sys.argv[1]
    
    return asyncio.run(process_image(image_path, database))


if __name__ == "__main__":
    sys.exit(main())
