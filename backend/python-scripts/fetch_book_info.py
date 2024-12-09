import requests
import json
import re
from typing import Dict, Any, Optional, Tuple, List

def fetch_book_data(title: str, author: str) -> Optional[str]:
    """
    Fetch book information from Open Library API using title and author.
    
    Args:
        title (str): Book title
        author (str): Book author
        
    Returns:
        Optional[str]: JSON string containing book data or None if not found
    """
    base_url = "https://openlibrary.org/search.json"
    params = {
        "title": title,
        "author": author
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('docs'):
            return None
            
        book_info = data['docs'][0]
        edition_keys = book_info.get('edition_key', [])[:10]  # Limit to 10 editions
        
        # Get series info from Audible
        series_name, series_position, series_type = get_series_info_from_audible_by_title_and_author(
            title, author, book_info.get('first_publish_year', '')
        )
        
        # Process contributors
        contributors = book_info.get("contributor", [])
        editors, illustrators, other_contributors = categorize_contributors(contributors)
        
        # Extract genres and subjects
        subjects = book_info.get('subject', [])
        genres, subjects, is_fiction, is_non_fiction, is_blended = extract_genre_and_subject_info(subjects)
        
        # Get editions information
        editions = []
        for key in edition_keys:
            edition_info = fetch_edition_info(key, author)
            if edition_info:
                editions.append(edition_info)
        
        # Construct the final book data
        book_data = {
            "title": book_info.get('title'),
            "subtitle": book_info.get('subtitle'),
            "creators": {
                "authors": book_info.get('author_name', []),
                "illustrators": illustrators,
                "editors": editors,
                "other_contributors": other_contributors
            },
            "copyright_date": book_info.get('first_publish_year'),
            "synopsis": editions[0].get('synopsis') if editions else None,
            "series": {
                "name": series_name,
                "position": series_position,
                "type": series_type
            },
            "genres": genres,
            "form": None,  # Not available from Open Library
            "format": None,  # Not available from Open Library
            "isbn_10": book_info.get('isbn', [None])[0] if book_info.get('isbn') else None,
            "isbn_13": book_info.get('isbn', [None, None])[1] if book_info.get('isbn') else None,
            "page_count": book_info.get('number_of_pages_median'),
            "book_type": "Fiction" if is_fiction else "Non-Fiction" if is_non_fiction else "Blended" if is_blended else None,
            "additional_info": {
                "subjects": subjects,
                "editions": editions,
                "publisher": book_info.get('publisher', []),
                "publish_date": book_info.get('publish_date', []),
                "languages": book_info.get('language', [])
            }
        }
        
        return json.dumps(book_data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching book data: {e}")
        return None
    except Exception as e:
        print(f"Error processing book data: {e}")
        return None

def fetch_edition_info(edition_key: str, author: str) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information for a specific edition from Open Library.
    
    Args:
        edition_key (str): Open Library edition key
        author (str): Book author
        
    Returns:
        Optional[Dict]: Dictionary containing edition information or None if not found
    """
    url = f"https://openlibrary.org/books/{edition_key}.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        edition = response.json()
        
        return {
            "title": edition.get("title"),
            "author": author,
            "isbn_10": edition.get("isbn_10", [None])[0] if edition.get("isbn_10") else None,
            "isbn_13": edition.get("isbn_13", [None])[0] if edition.get("isbn_13") else None,
            "image": f"https://covers.openlibrary.org/b/id/{edition.get('covers', [])[0]}-L.jpg" if edition.get("covers") else None,
            "images": [f"https://covers.openlibrary.org/b/id/{cover}-L.jpg" for cover in edition.get("covers", [])],
            "page_count": edition.get("number_of_pages"),
            "publish_date": edition.get("publish_date"),
            "synopsis": edition.get("description", {}).get("value") if isinstance(edition.get("description"), dict) else edition.get("description")
        }
    except Exception as e:
        print(f"Error fetching edition info: {e}")
        return None

def get_series_info_from_audible_by_title_and_author(
    title: str, 
    author: str, 
    pub_date: str
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Fetch series information from Audible API.
    
    Args:
        title (str): Book title
        author (str): Book author
        pub_date (str): Publication date
        
    Returns:
        Tuple[Optional[str], Optional[str], str]: Series name, position, and type
    """
    url = "https://api.audible.com/1.0/catalog/products"
    params = {
        "title": title,
        "author": author,
        "release_date": pub_date,
        "num_results": 1,
        "response_groups": "series"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        book_data = response.json()
        products = book_data.get("products", [])
        
        if not products:
            return None, None, "Stand Alone"
            
        series_info = products[0].get("series", [])
        if series_info:
            series_name = series_info[0].get("title")
            series_position = series_info[0].get("sequence")
            return series_name, series_position, "Series"
            
        return None, None, "Stand Alone"
        
    except Exception as e:
        print(f"Error fetching series info: {e}")
        return None, None, "Stand Alone"

def categorize_contributors(contributors: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize contributors into editors, illustrators, and others.
    
    Args:
        contributors (List[str]): List of contributor strings
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Lists of editors, illustrators, and other contributors
    """
    editors = []
    illustrators = []
    other_contributors = []
    
    patterns = {
        "editor": re.compile(r"\(Editor\)", re.IGNORECASE),
        "illustrator": re.compile(r"\(Illustrator\)", re.IGNORECASE),
        "translator": re.compile(r"\btranslator\b", re.IGNORECASE),
        "narrator": re.compile(r"\bNarrator\b", re.IGNORECASE)
    }
    
    for contributor in contributors:
        if patterns["editor"].search(contributor):
            editors.append(patterns["editor"].sub("", contributor).strip())
        elif patterns["illustrator"].search(contributor):
            illustrators.append(patterns["illustrator"].sub("", contributor).strip())
        elif patterns["translator"].search(contributor) or patterns["narrator"].search(contributor):
            other_contributors.append(contributor)
        else:
            other_contributors.append(contributor)
            
    return editors, illustrators, other_contributors

def extract_genre_and_subject_info(subject_list: List[str]) -> Tuple[Dict[str, List[str]], List[str], bool, bool, bool]:
    """
    Extract and categorize genres and subjects from a list of subjects.
    
    Args:
        subject_list (List[str]): List of subject strings
        
    Returns:
        Tuple[Dict[str, List[str]], List[str], bool, bool, bool]: 
            Genres dictionary, subjects list, and fiction/non-fiction/blended flags
    """
    genres = {
        "fiction": [],
        "non_fiction": []
    }
    subjects = []
    
    is_fiction = False
    is_non_fiction = False
    
    for subject in subject_list:
        subject_lower = subject.lower()
        if "fiction" in subject_lower:
            genres["fiction"].append(subject)
            is_fiction = True
        elif "non-fiction" in subject_lower:
            genres["non_fiction"].append(subject)
            is_non_fiction = True
        else:
            subjects.append(subject)
    
    is_blended = is_fiction and is_non_fiction
    
    return genres, subjects, is_fiction, is_non_fiction, is_blended

def fetch_book_data_by_isbn(isbn: str) -> Optional[str]:
    """
    Fetch book information from Open Library API using ISBN.
    
    Args:
        isbn (str): ISBN-10 or ISBN-13
        
    Returns:
        Optional[str]: JSON string containing book data or None if not found
    """
    # Clean ISBN
    isbn = isbn.replace('-', '').replace(' ', '')
    
    # Try Open Library's ISBN API first
    base_url = f"https://openlibrary.org/api/books"
    params = {
        "bibkeys": f"ISBN:{isbn}",
        "format": "json",
        "jscmd": "data"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None
            
        book_data = data.get(f"ISBN:{isbn}")
        if not book_data:
            return None
            
        # Get series info if authors are available
        series_name, series_position, series_type = None, None, "Stand Alone"
        if book_data.get('authors'):
            author_name = book_data['authors'][0].get('name', '')
            series_name, series_position, series_type = get_series_info_from_audible_by_title_and_author(
                book_data.get('title', ''),
                author_name,
                book_data.get('publish_date', '')
            )
        
        # Extract subjects and genres
        subjects = book_data.get('subjects', [])
        genres, filtered_subjects, is_fiction, is_non_fiction, is_blended = extract_genre_and_subject_info(
            [s.get('name', '') for s in subjects] if isinstance(subjects[0], dict) else subjects
        )
        
        # Process contributors if available
        contributors = book_data.get('contributors', [])
        editors, illustrators, other_contributors = categorize_contributors(
            [f"{c.get('name', '')} ({c.get('role', '')})" for c in contributors] if contributors else []
        )
        
        # Construct the final book data
        formatted_data = {
            "title": book_data.get('title'),
            "subtitle": book_data.get('subtitle'),
            "creators": {
                "authors": [author.get('name') for author in book_data.get('authors', [])],
                "illustrators": illustrators,
                "editors": editors,
                "other_contributors": other_contributors
            },
            "copyright_date": book_data.get('publish_date'),
            "synopsis": book_data.get('description'),
            "series": {
                "name": series_name,
                "position": series_position,
                "type": series_type
            },
            "genres": genres,
            "form": None,  # Try to determine from subjects/classification
            "format": book_data.get('physical_format'),
            "isbn_10": book_data.get('identifiers', {}).get('isbn_10', [None])[0],
            "isbn_13": book_data.get('identifiers', {}).get('isbn_13', [None])[0],
            "page_count": book_data.get('number_of_pages'),
            "book_type": "Fiction" if is_fiction else "Non-Fiction" if is_non_fiction else "Blended" if is_blended else None,
            "additional_info": {
                "subjects": filtered_subjects,
                "publisher": book_data.get('publishers', []),
                "publish_places": book_data.get('publish_places', []),
                "languages": [lang.get('key', '').split('/')[-1] for lang in book_data.get('languages', [])],
                "covers": book_data.get('cover', {})
            }
        }
        
        return json.dumps(formatted_data)
        
    except requests.exceptions.RequestException as e:
        import traceback
        print(traceback.format_exc())
        print(f"Error fetching book data by ISBN: {e}")
        return None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Error processing book data by ISBN: {e}")
        return None
