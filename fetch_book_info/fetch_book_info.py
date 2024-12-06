import requests
import json
import re


def fetch_edition_info(edition_key, author):
    url = f"https://openlibrary.org/books/{edition_key}.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": f"Failed to fetch data for edition key {edition_key}"}
    
    edition = response.json()
    # print(json.dumps(edition, indent=4))
    
    with open("editions_info_from_openlibrary.json", "w", encoding="utf-8") as f:
        json.dump(edition, f, ensure_ascii=False, indent=4)
    # Extracting information based on your desired format
    edition_info = {
        # # Edition key
        # "editionId": edition.get("key", "N/A"),
        
        "title": edition.get("title", "N/A"),
        "author": author, # no author name in the fetched info just the author key
        
        "isbn10": edition.get("isbn_10", "N/A"),
        "isbn13": edition.get("isbn_13", "N/A"),
    
        "image": f"https://covers.openlibrary.org/b/id/{edition.get('covers', [])[0]}-L.jpg" if edition.get("covers") else "N/A",
        "images": [f"https://covers.openlibrary.org/b/id/{cover}-L.jpg" for cover in edition.get("covers", [])] or ["N/A"],
        "pageCount": edition.get("number_of_pages", "N/A"),
        "wordCount": 0, # not available
        "pubDate": edition.get("publish_date", "N/A"),
        "copyrightDate": edition.get("publish_date", "N/A"),
        "synopsis": edition.get("description", {}).get("value", "N/A") if isinstance(edition.get("description"), dict) else edition.get("description", "N/A"),
        "format": "N/A", # not available
        "isUnpaged": False
    }
    
    return edition_info

def get_series_info_from_audible_by_title_and_author(title, author, pub_date):
    url = "https://api.audible.com/1.0/catalog/products"

    params = {
        "title": title,
        "author": author,
        "release date": pub_date,
        "num_results": 1,  # Limit to 1 result for precision
        "response_groups": "series"  # Get series and contributors information
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        book_data = response.json()
        # print(json.dumps(book_data, indent=4))
        # Check if 'products' is in the response and has content
        products = book_data.get("products", [])
        if not products:
            # print("No book found for the given title and author.")
            return "N/A", "N/A", "Stand Alone"  # No books found

        # If products exist, get the first one
        product_info = products[0]
        series_info = product_info.get("series", [])
        # print(product_info)
            

        if series_info:
            # Extract and return series name and position
            series_name = series_info[0].get("title", "Unknown Series")
            series_position = series_info[0].get("sequence", "Unknown Position")
            return series_name, series_position, "Series"
        else:
            # print("The book is not part of any series.")
            return "N/A", "N/A", "Stand Alone"

    except requests.exceptions.RequestException as e:
        # Handle network or HTTP errors
        print(f"Error fetching series information: {e}")
        return None

    except ValueError as ve:
        # Handle cases where the response is not valid JSON
        print(f"Error parsing the response: {ve}")
        return None



def extract_genre_and_subject_info(subject_list):

    genres = {
        "fiction": [],
        "non_fiction": []
    }
    subjects = []

    # Flags for fiction and non-fiction
    is_fiction = False
    is_non_fiction = False

    # Categorize each subject
    for subject in subject_list:
        if "fiction" in subject.lower():
            genres["fiction"].append(subject)
            is_fiction = True
        elif "non-fiction" in subject.lower():
            genres["non_fiction"].append(subject)
            is_non_fiction = True
        else:
            subjects.append(subject)

    # Determine if the genres are blended (both fiction and non-fiction)
    is_blended = is_fiction and is_non_fiction

    return genres,subjects, is_fiction,is_non_fiction,is_blended
def categorize_contributors(contributors):
    # Define lists to store the categorized contributors
    editors = []
    illustrators = []
    other_contributors = []

    # Define regex patterns for matching roles
    editor_pattern = re.compile(r"\(Editor\)", re.IGNORECASE)
    illustrator_pattern = re.compile(r"\(Illustrator\)", re.IGNORECASE)
    translator_pattern = re.compile(r"\btranslator\b", re.IGNORECASE)
    narrator_pattern = re.compile(r"\bNarrator\b", re.IGNORECASE)

    # Iterate through the contributors and categorize them
    for contributor in contributors:
        if editor_pattern.search(contributor):
            editor_name = editor_pattern.sub("", contributor).strip()
            editors.append(editor_name)
        elif illustrator_pattern.search(contributor):
            illustrator_name = illustrator_pattern.sub("", contributor).strip()
            illustrators.append(illustrator_name)
        elif translator_pattern.search(contributor) or narrator_pattern.search(contributor):
            other_contributors.append(contributor)
        else:
            # If no match, add them to 'Other Contributors'
            other_contributors.append(contributor)

    return editors, illustrators, other_contributors


# use Open Library API to handle different editions
# but it doesn't have description field
def fetch_book_data(title, author):
    base_url = "https://openlibrary.org/search.json" # partial and fuzzy search
    params = {
        "title": title,
        "author": author
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    # Write results to a JSON file
    with open("singlebookinfo_from_open_library.json", "w", encoding="utf-8") as f:
        json.dump(data['docs'][0], f, ensure_ascii=False, indent=4)

    # Check for main book data
    book_info = data['docs'][0] if data['docs'] else {}
    editions = []
    contributors = book_info.get("contributor", [])
    editors, illustrators, other_contributors = categorize_contributors(contributors)
    # isbn10 = next((isbn for isbn in book_info.get("isbn", []) if len(isbn) == 10), "N/A"),
    # isbn13 = next((isbn for isbn in book_info.get("isbn", []) if len(isbn) == 13), "N/A"),
    isbns = book_info.get("isbn")
    # print(isbns)
    # description = fetch_description_from_google_by_isbn(isbn10, isbn13)
    copyright_date = book_info.get("first_publish_year", "N/A")
    asins = book_info.get("id_amazon")
    # print(asins)
    series, seriesBookNumber, seriesType = get_series_info_from_audible_by_title_and_author(title, author, copyright_date)
    subject_list = book_info.get("subject", [])
    genre, subjects, isFiction, isNonFiction, isBlended = extract_genre_and_subject_info(subject_list)
    edition_keys = book_info.get("edition_key")

    # Define JSON structure for Must Haves
    must_haves = {
        "title": book_info.get("title", "N/A"),
        "subtitle": book_info.get("subtitle", "N/A"),
        "authors": book_info.get("author_name", ["N/A"]),
        "editors": editors,
        "illustrators": illustrators,
        "Other Contributors": other_contributors,

        "copyRightDate": book_info.get("first_publish_year", "N/A"),
        # "synopsis": description,

        "series": series,
        "seriesBookNumber": seriesBookNumber,
        "seriesType": seriesType,

        "genre": genre,
        "narrativeForm": "N/A", 

        "format": book_info.get("format", []),

        # "isbn10": isbn10,
        # "isbn13": isbn13,
        "isbns": isbns,
        "pageCount": book_info.get("number_of_pages_median", "N/A"),

        "isFiction": isFiction,
        "isNonFiction": isNonFiction,
        "isBlended": isBlended,
    }

    # Define JSON structure for Optional fields
    optional = {
        "publisher": book_info.get("publisher", ["N/A"])[0] if book_info.get("publisher") else "N/A",
        "pubDate": book_info.get("first_publish_year", "N/A"),
        "subgenre": "N/A",
        "internationalAwards": "N/A",
        "guidedReadingLevel": "N/A",
        "lexileLevel": "N/A",
        "textFeatures": "N/A"
    }

    # Define JSON structure for Extras (Nice to Have) fields
    extras = {
        # "topic": subjects,
        "subject": subjects,
        "tags": "N/A",
        "targetAudience": "N/A",
        "bannedBookFlag": False,
        "alternateTitles": "N/A",
        "images": book_info.get("cover_i", "N/A"),
        "voice": "N/A"
    }

    # Handle Editions
    # to ensure effiency, only 10 of them
    num_of_edition = 10
    for i in range(0, num_of_edition):
        key = edition_keys[i]
        edition_info = fetch_edition_info(key, author)
        editions.append(edition_info)

    # Collect all data in the final JSON structure with flags for missing data
    book_json = {
        **must_haves,
        **optional,
        **extras,
        "editions": editions,
        "flags": {
            "duplication": False,
            "collisions": False,
            "missing_data": [key for key, value in {**must_haves, **optional, **extras}.items() if
                             value == "N/A"]
        }
    }

    # Convert to JSON string for readability
    book_json_str = json.dumps(book_json, indent=4)
    return book_json_str

# Example call
title = "Harry Potter and the Chamber of Secrets"
author = "J. K. Rowling"
book_data = fetch_book_data(title, author)
print(book_data)


