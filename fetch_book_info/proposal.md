0) How are you measuring your results?

Lulu: to measure the completeness of require fields of book info

1) What you tried so far
Lulu:
- to ensure the completeness of require fields. I tried different APIs including OpenLibrary API, Audible API, Google Books API, goodreads API

2) what worked
Lulu: 
    - to handle editions, Open Library API work perfectly  all editions key are included in fetched info, then call fetch edition info by edition key
    - to get series information, audible API include the seires infomation. we fetch series info by author and title
3) what did not work
    - Goodreads API don't issue public keys for API access anymore
    - google API don't work for editions because the original book is not related to other editions.
    - i tried to use fetch series information by Amazon id fetched from openlibrary API, but either there is no products related to that id, either the products is not the book we're looking for
4) why did/did not work
    -  If we tried to include the few most relevant books, we can't decide whether they're the different versions of the same book
    - we can't fetch info by amazon id, might be the accuracy of info in open library database, or the product infomation is outdated, or the inconsistency of amazon database


5) what are you going to do moving forward.
    - to make sure the info fetched from audible API matches with the book in open library, we used publish date to refine the result, but it's not ideal because the book can have varied publish date and the release date of audio book might not match with the physical ones. we need to find a better way
    - try to find the as many as required fields.
