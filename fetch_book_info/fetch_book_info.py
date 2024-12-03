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

# don't need authorization
# Step 3: Use the audile API to fetch series info
# don't support isbn search
# only limited to the products of audio book
# asin might be included in the info fetched from open library API, but it didn't work? the fetched product info doesn't match

# def get_series_info_from_audible(asin_list):
#     """
#     Fetch series information from Audible by iterating over ASINs in the list.
#     Exits the loop as soon as valid series information is found.
#     """
#     url_asin = "https://api.audible.com/1.0/catalog/products"

#     # Iterate over each ASIN in the list
#     if asin_list:
#         for asin in asin_list:
#             try:
#                 response = requests.get(f"{url_asin}/{asin}", params={"response_groups": "series"})
#                 response.raise_for_status()
#                 response_info = response.json()
                
#                 # Check if 'products' is in the response and has content
#                 products = response_info.get("products", [])
#                 if not products:
#                     continue
#                 # If products exist, get the first one
#                 product_info = products[0]
#                 series_info = product_info.get("series", [])
#                 print("product info" + asin)
#                 print(response_info)
#                 if series_info:
#                     # Extract and return series name and position
#                     series_name = series_info[0].get("title", "N/A")
#                     series_position = series_info[0].get("sequence", "N/A")
#                     return series_name, series_position, "Series"
#             except requests.exceptions.RequestException as e:
#                 print(f"Error fetching series information by ASIN {asin}: {e}")
#                 continue  # Try the next ASIN
#             except ValueError as ve:
#                 print(f"Error parsing response for ASIN {asin}: {ve}")
#                 continue  # Try the next ASIN

#     # If no ASIN found with series info, return standalone status
#     return "N/A", "N/A", "Stand Alone"

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

# don't need anymore because description can be acquired from edition_info
# # get book description from Google Book API, cause it is unavailable from OPEN LIBRARY
# def fetch_description_from_google_by_isbn(isbn10, isbn13):
#     # Check which ISBN is available and choose the correct one
#     isbn = isbn13 if isbn13 != "N/A" else isbn10

#     if isbn == "N/A":
#         return "N/A"

#     # Google Books API URL with ISBN query (using the selected ISBN)
#     url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"

#     try:
#         # Make the GET request to Google Books API
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors

#         # Attempt to parse the JSON response
#         try:
#             book_data = response.json()
#         except requests.exceptions.JSONDecodeError as e:
#             return f"Error parsing JSON response: {e}"

#         # Check if there are any items in the response
#         if "items" in book_data:
#             book_info = book_data["items"][0]["volumeInfo"]
#             description = book_info.get("description", "N/A (not available)")
#             return description
#         else:
#             return "N/A"

#     except requests.exceptions.RequestException as e:
#         return f"Error fetching data: {e}"


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
        # edition_info = {
        #     # "editionId": edition.get("key", "N/A"),
        #     "title": edition.get("title", "N/A"),
        #     # "subtitle": edition.get("subtitle", "N/A"),
        #     "author": ", ".join(edition.get("author_name", [])),
        #     # "foreword": "N/A",
        #     # "editor": "N/A",
        #     # "illustrator": "N/A",
        #     "isbn": {
        #         "isbn10": next((isbn for isbn in edition.get("isbn", []) if len(isbn) == 10), "N/A"),
        #         "isbn13": next((isbn for isbn in edition.get("isbn", []) if len(isbn) == 13), "N/A")
        #     },
        #     "image": f"https://covers.openlibrary.org/b/id/{edition.get('cover_i', '')}-L.jpg" if edition.get("cover_i") else "N/A",
        #     "images": [f"https://covers.openlibrary.org/b/id/{edition.get('cover_i', '')}-L.jpg"] if edition.get("cover_i") else ["N/A"],
        #     "pageCount": edition.get("number_of_pages_median", "N/A"),
        #     "wordCount": 0,
        #     "pubDate": edition.get("first_publish_year", "N/A"),
        #     "copyrightDate": edition.get("first_publish_year", "N/A"),
        #     "synopsis": edition.get("description", "N/A"),
        #     "format": "N/A",
        #     "isUnpaged": False
        # }
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


#
# # Google Book API doesn't support different versions of the same book, it just do the relevance search
# def fetch_book_data(title, author):
#     api_url = "https://www.googleapis.com/books/v1/volumes"
#     # Partial Matches: The Google Books API performs a case-insensitive, partial match, so even a part of the title or author text can yield results.
#     # Fuzzy Matching: The API uses fuzzy matching, so it can match similar words or close variations of the search terms.
#     # Relevance-Based Ranking: Results are ranked by relevance, so the first item(s) returned in the results should be the best matches based on the query.
#     params = {
#         'q': f'intitle:{title}+inauthor:{author}',
#         'printType': 'books',
#         'maxResults': 1
#     }
#     response = requests.get(api_url, params=params)
#
#     if response.status_code != 200:
#         print("Error fetching data from API.")
#         return None
#
#     data = response.json()
#
#     if "items" not in data or not data["items"]:
#         print("No results found.")
#         return None
#
#     book_info = data["items"][0]["volumeInfo"]
#
#     # Define JSON structure for Must Haves
#     must_haves = {
#         # Tile and Subtitle
#         "title": book_info.get("title", "N/A"),
#         "subtitle": "N/A (not available)", # NULLABLE
#
#         # Creators(author, illustrators, editor, etc.)
#         "author": author,
#         "editor": "N/A (not available)",  # NULLABLE
#         "illustrator": "N/A (not available)",  # NULLABLE
#         "creators": {
#             "Other Contributors": book_info.get("contributors", [])
#         },
#         "copyRightDate": book_info.get("publishedDate", "N/A"),
#         "synopsis": book_info.get("description", "N/A"),
#         # Series Name and Position (#1, #2, etc.)
#         "series": "N/A", #NULLABLE
#         "seriesBookNumber": "1", # DEFAULT TO 1
#         "seriesType": "Stand Alone", # DEFAULT TO Stand Alone, Not in Google Books API
#
#         "genre": book_info.get("categories", []),
#         # Form (e.g., Graphic Novel, Picturebook)
#         "narrativeForm": book_info.get("printType", "N/A"), # Not in Google Books API
#         # Format (e.g., Audiobook, Paperback, etc.)
#         "format": "N/A (not available)",  # Not in Google Books API
#         # ISBN# both
#         "isbn10": next((identifier["identifier"] for identifier in book_info.get("industryIdentifiers", []) if
#                       identifier["type"].startswith("ISBN")), "N/A"),
#         "isbn13": next((identifier["identifier"] for identifier in book_info.get("industryIdentifiers", []) if
#                       identifier["type"].startswith("ISBN")), "N/A"),
#         "pageCount": book_info.get("pageCount", "N/A"),
#         # Type of book(Fiction, Nonfiction, or Blended)
#         "isFiction": True if "fiction" in (book_info.get("categories", [""])[0].lower()) else False,
#         "isNonFiction": False if "fiction" in (book_info.get("categories", [""])[0].lower()) else True,
#         "isBlended": False,
#     }
#
#     # Define JSON structure for Optional fields
#     optional = {
#         "publisher": book_info.get("publisher", "N/A"),
#         "pubDate": book_info.get("publishedDate", "N/A"),
#         "subgenre": "N/A (not available)",  # Not in Google Books API
#         "internationalAwards": "N/A (not available)",  # Not in Google Books API
#         "guidedReadingLevel": "N/A (not available)",  # Not in Google Books API
#         "lexileLevel": "N/A (not available)",  # Not in Google Books API
#         # Text features(e.g., Table of Contents, Sources)
#         "textFeatures": "N/A (not available)"  # Not in Google Books API
#     }
#
#     # Define JSON structure for Extras (Nice to Have) fields
#     extras = {
#         "topic": book_info.get("categories", []), # Partially available through volumeInfo.categories or keywords in volumeInfo.description
#         "subject": book_info.get("categories", []), # Limited subject classification through volumeInfo.categories
#         "tags": "N/A (not available)", # Not in Google Books API
#         "targetAudience": book_info.get("maturityRating", "N/A"),
#         "bannedBookFlag": False,  # Placeholder, requires external data
#         "alternateTitles": "N/A (not available)",  # Not in Google Books API
#         # covers
#         "images": book_info.get("imageLinks", {}).get("thumbnail", "N/A"),
#         "voice": "N/A (not available)"
#     }
#
#     # Handle Editions
#     editions = []
#     if 'items' in data:
#         for item in data['items']:
#             edition_info = item.get("volumeInfo", {})
#             edition = {
#                 "editionId": item.get("id", "N/A"),
#                 "title": edition_info.get("title", "N/A"),
#                 "subtitle": edition_info.get("subtitle", "N/A"),
#                 "author": ", ".join(edition_info.get("authors", [])),
#                 "foreword": "N/A",
#                 "editor": "N/A",
#                 "illustrator": "N/A",
#                 "isbn": {
#                     "isbn10": next((identifier["identifier"] for identifier in edition_info.get("industryIdentifiers", []) if identifier["type"] == "ISBN_10"), "N/A"),
#                     "isbn13": next((identifier["identifier"] for identifier in edition_info.get("industryIdentifiers", []) if identifier["type"] == "ISBN_13"), "N/A")
#                 },
#                 "image": edition_info.get("imageLinks", {}).get("thumbnail", "N/A"),
#                 "images": [edition_info.get("imageLinks", {}).get("thumbnail", "N/A")],
#                 "pageCount": edition_info.get("pageCount", "N/A"),
#                 "wordCount": 0, # Placeholder if no word count data is available
#                 "pubDate": edition_info.get("publishedDate", "N/A"),
#                 "copyrightDate": edition_info.get("publishedDate", "N/A"),
#                 "synopsis": edition_info.get("description", "N/A"),
#                 "format": edition_info.get("printType", "N/A"),
#                 "isUnpaged": False # Placeholder
#             }
#             editions.append(edition)
#
#     # Collect all data in the final JSON structure with flags for missing data
#     book_json = {
#         **must_haves,
#         **optional,
#         **extras,
#         "editions": editions,
#         "flags": {
#             "duplication": False,
#             "collisions": False,
#             "missing_data": [key for key, value in {**must_haves, **optional, **extras}.items() if
#                              value == "N/A (not available)"]
#         }
#     }
#
#     # Convert to JSON string for readability
#     book_json_str = json.dumps(book_json, indent=4)
#     return book_json_str
#
#
# # Example usage
# title = "To Kill a Mockingbird"
# author = "Harper Lee"
# book_data_json = fetch_book_data(title, author)
#
# if book_data_json:
#     print(book_data_json)
