// Check if script is loaded
console.log('JavaScript file loaded');

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM fully loaded');
  
  // Get button element
  const uploadButton = document.getElementById('uploadButton');
  if (uploadButton) {
    console.log('Upload button found');
    // Add click event listener
    uploadButton.addEventListener('click', (event) => uploadImage(event));
  } else {
    console.error('Upload button not found');
  }
});

// Function to handle the image upload
function uploadImage(event) {
  // Prevent default form submission if this is called from a form
  event.preventDefault();
  
  console.log('Upload function called');
  
  const fileInput = document.getElementById("imageInput")
  const file = fileInput.files[0]

  console.log("Selected file:", file)

  if (!file) {
    alert("Please select an image first!")
    return
  }

  const formData = new FormData()
  formData.append("image", file)

  // Show loading state
  const resultDiv = document.getElementById("result")
  resultDiv.textContent = "Processing..."

  console.log("Sending request to server...")
  fetch("http://localhost:3000/upload", {
    method: "POST",
    body: formData,
    credentials: 'include',
    mode: 'cors'
  })
    .then((response) => {
      console.log("Server response status:", response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Server response data:", data)
      
      // Check if the request was successful
      if (data.success && data.result) {
        console.log("Number of books detected:", data.result.length)
        resultDiv.innerHTML = formatResults(data.result, data.annotatedImage)
      } else {
        console.error("Server returned error:", data.error)
        resultDiv.textContent = data.error || "Error processing image"
      }
    })
    .catch((error) => {
      console.error("Fetch error:", error)
      resultDiv.textContent = "Error processing image"
    })
}

function formatResults(books, annotatedImageBase64) {
  console.log("Formatting books:", books)
  
  let html = '';
  
  if (annotatedImageBase64) {
    html += `
      <div class="annotated-image mb-4">
        <h3>Detected Books:</h3>
        <img src="${annotatedImageBase64}" alt="Annotated books" class="img-fluid rounded shadow">
      </div>
    `;
  }
  
  html += '<div class="scrollable-results">';
  
  if (!Array.isArray(books) || books.length === 0) {
    html += "No books detected";
  } else {
    html += books.map((book, index) => {
      return `
        <div class="book-result">
          <h3>Book ${index + 1}</h3>
          
          <div class="row">
            <div class="col${book.cover_image ? '-8' : ''}">
              <div class="book-info">
                <h4>Must Have Information</h4>
                <p><strong>Title:</strong> ${book.title || 'Unknown'}</p>
                <p><strong>Subtitle:</strong> ${book.subtitle || 'N/A'}</p>
                
                <div class="creators">
                  <p><strong>Authors:</strong> ${Array.isArray(book.creators?.authors) ? book.creators.authors.join(', ') : 'Unknown'}</p>
                  <p><strong>Illustrators:</strong> ${Array.isArray(book.creators?.illustrators) ? book.creators.illustrators.join(', ') : 'N/A'}</p>
                  <p><strong>Editors:</strong> ${Array.isArray(book.creators?.editors) ? book.creators.editors.join(', ') : 'N/A'}</p>
                </div>

                <p><strong>Copyright Date:</strong> ${book.copyright_date || 'N/A'}</p>
                <p><strong>Summary:</strong> ${book.summary || 'N/A'}</p>
                
                <div class="series-info">
                  <p><strong>Series:</strong> ${book.series?.name || 'N/A'}</p>
                  <p><strong>Position in Series:</strong> ${book.series?.position || 'N/A'}</p>
                </div>

                <p><strong>Genres:</strong> ${Array.isArray(book.genres) ? book.genres.join(', ') : 'N/A'}</p>
                <p><strong>Form:</strong> ${book.form || 'N/A'}</p>
                <p><strong>Format:</strong> ${book.format || 'N/A'}</p>
                <p><strong>ISBN-10:</strong> ${book.isbn_10 || 'N/A'}</p>
                <p><strong>ISBN-13:</strong> ${book.isbn_13 || 'N/A'}</p>
                <p><strong>Page Count:</strong> ${book.page_count || 'N/A'}</p>
                <p><strong>Book Type:</strong> ${book.book_type || 'N/A'}</p>
              </div>
            </div>
            
            ${book.cover_image ? `
              <div class="col-4">
                <div class="book-cover h-100 d-flex align-items-start">
                  <img src="${book.cover_image}" 
                       alt="Cover of ${book.title}" 
                       class="img-fluid rounded shadow" 
                       style="max-height: 300px; object-fit: contain;">
                </div>
              </div>
            ` : ''}
          </div>

          <div class="source-info mt-3">
            <p><strong>Data Source:</strong> ${book.source || 'Unknown'}</p>
            ${book.matchScore ? `<p><strong>Match Score:</strong> ${book.matchScore}%</p>` : ''}
          </div>

          ${book.detectedText ? `
            <div class="detected-text mt-3">
              <p><strong>Detected Text:</strong></p>
              <pre class="text-muted">${book.detectedText}</pre>
              <div class="original-detection mt-2">
                <p><strong>Originally Detected:</strong></p>
                <p class="text-muted mb-1">Title: ${book.original_title || 'N/A'}</p>
                <p class="text-muted">Author: ${book.original_author || 'N/A'}</p>
              </div>
            </div>
          ` : ''}
        </div>
      `;
    }).join('<hr>')
  }
  
  html += '</div>';
  return html;
}
