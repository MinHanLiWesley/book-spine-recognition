### Must Haves

1. **Title & Subtitle**
   - JSON Field: `"title"`, `"subtitle"`
   - Google Books API: `volumeInfo.title` (often includes subtitle in the main title if no separate field is available)
2. **Creators (author, illustrators, editor, etc.)**
   - JSON Fields: `"author"`, `"editor"`, `"illustrator"`, `"creators"`
   - Google Books API: `volumeInfo.authors` (authors are provided; additional creators like editors may not always be available)
3. **Copyright date**
   - JSON Field: `"editions[0].copyrightDate"`
   - Google Books API: Not specifically available; sometimes inferred from `volumeInfo.publishedDate`
4. **Summary/Synopsis**
   - JSON Field: `"synopsis"` or `"editions[0].synopsis"`
   - Google Books API: `volumeInfo.description` (summary available)
5. **Series Name and Position (#1, #2, etc.)**
   - JSON Fields: `"series"`, `"seriesBookNumber"`
   - Google Books API: Not typically available; series data is limited and may need to be sourced externally
6. **Genres**
   - JSON Field: `"genre"`
   - Google Books API: `volumeInfo.categories` (provides broad genres)
7. **Form (e.g., Graphic Novel, Picturebook)**
   - JSON Field: `"narrativeForm"`
   - Google Books API: Not directly available; sometimes inferred from `volumeInfo.categories` or `volumeInfo.description`
8. **Format (e.g., Audiobook, Paperback, etc.)**
   - JSON Field: `"editions[0].format"`
   - Google Books API: Not directly available; format may need to be cross-referenced with editions info or from marketplace data
9. **ISBN #**
   - JSON Fields: `"isbn10"`, `"isbn13"`
   - Google Books API: `volumeInfo.industryIdentifiers` (provides both ISBN-10 and ISBN-13)
10. **# of pages**
    - JSON Field: `"editions[0].pageCount"`
    - Google Books API: `volumeInfo.pageCount` (number of pages provided for physical formats)
11. **Type of book (Fiction, Nonfiction, or Blended)**
    - JSON Fields: `"isFiction"`, `"isNonFiction"`, `"isBlended"`
    - Google Books API: Not directly available; genre categories may hint at fiction/nonfiction but often requires external confirmation

### Optional

1. **Publisher**
   - JSON Field: `"publisher"`
   - Google Books API: `volumeInfo.publisher` (publisher information provided)
2. **Publication Date**
   - JSON Field: `"pubDate"` or `"editions[0].pubDate"`
   - Google Books API: `volumeInfo.publishedDate` (publication date provided)
3. **Subgenres**
   - JSON Field: Can be inferred from `"genre"`
   - Google Books API: Limited subgenre information, sometimes inferred from `volumeInfo.categories`
4. **Awards**
   - JSON Fields: `"awards"`, `"internationalAwards"`
   - Google Books API: Not available; awards data typically requires external sources
5. **Guided Reading Level/Lexile Level**
   - JSON Fields: `"guidedReadingLevel"`, `"lexileLevel"`
   - Google Books API: Not available
6. **Text features (e.g., Table of Contents, Sources)**
   - JSON Field: `"textFeatures"`
   - Google Books API: Not directly available; some text features may be inferred from `volumeInfo.description`

### Extras (Nice to Have)

1. **Topics**
   - JSON Field: `"topic"`
   - Google Books API: Partially available through `volumeInfo.categories` or keywords in `volumeInfo.description`
2. **Subjects**
   - JSON Field: `"subject"`
   - Google Books API: Limited subject classification through `volumeInfo.categories`
3. **Target audience**
   - JSON Field: Derived from tags like `"voice"` or `"tags"`
   - Google Books API: Not directly available; some hint through `volumeInfo.categories`
4. **Banned Book flag**
   - JSON Field: Not available
   - Google Books API: Not available
5. **Alternate Titles (e.g., UK vs USA titles)**
   - JSON Field: `"editions[].title"`
   - Google Books API: Not directly available; alternate titles need to be sourced from multiple editions if provided
6. **Covers**
   - JSON Field: `"editions[0].images"` and `"images"`
   - Google Books API: `volumeInfo.imageLinks` (provides URLs for thumbnail and small cover images)