# Book Spine Detection System

An AI-powered system that detects and extracts metadata from book spines in images. The system uses computer vision and machine learning to identify books, read their spines, and extract title and author information.

## Features

- Book spine detection using YOLO object detection
- Image enhancement using RealESRGAN
- Text extraction using Google Cloud Vision API
- Metadata refinement using Google Gemini AI
- Intelligent caching system to reduce API costs
- Web interface for uploading and viewing results

## System Requirements

### Backend Requirements
- Python 3.8+
- CUDA-capable GPU (for YOLO detection)
- Node.js and npm

### Python Dependencies

- torch
- torchvision
- opencv-python
- numpy
- pillow
- requests
- google-cloud-vision
- google-cloud-gemini


### External Models and APIs Required
- YOLO weights file (`models/yolo_weights/best.pt`)
  - Download the weights file from [Google Drive](https://drive.google.com/file/d/1zIxEQuPFWKhE7PxC4Ry5vJEnAI1MWOFB/view?usp=sharing)
  - Place the downloaded `best.pt` file in `models/yolo_weights/` directory
- RealESRGAN executable (`models/realesrgan_portable/realesrgan-ncnn-vulkan.exe`)
  - Download the portable executable from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
  - For Windows: Use realesrgan-ncnn-vulkan.exe
  - For Mac/Linux: Download appropriate version and adjust path accordingly
- Google Cloud Vision API credentials
- Google Gemini API key

### RealESRGAN Setup
1. Download the portable RealESRGAN executable for your platform
2. Place the executable in `models/realesrgan_portable/`
3. The system uses RealESRGAN with these default settings:
   ```bash
   # Windows example
   realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrgan-x4plus
   ```
   Available models:
   - realesrgan-x4plus (default)
   - realesrnet-x4plus
   - realesrgan-x4plus-anime (optimized for anime images)
   - realesr-animevideov3 (animation video)

Note: For Mac/Linux users, adjust the executable path and filename according to your platform.

## API Setup Requirements

### Google Cloud Vision API
1. Create a Google Cloud Project
2. Enable the Cloud Vision API
3. Create service account credentials
4. Download the JSON key file
5. Set up authentication by either:
   - Setting the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your key file:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
     ```
   - Or placing the JSON key file in a known location and updating the code to reference it

### Google Gemini API
1. Get a Gemini API key from Google AI Studio
2. Create a `.env` file in the backend directory
3. Add your Gemini API key:
   ```bash
   GEMINI_API=your_api_key_here
   ```

The system uses these APIs for:
- Google Cloud Vision: Text extraction from book spines
- Google Gemini: Intelligent refinement of extracted text and metadata parsing

For detailed Google Cloud Vision setup instructions, visit the [official documentation](https://cloud.google.com/vision/docs/setup).

## Project Structure
```
├── backend/
│ ├── python-scripts/
│ │ ├── detect.py # Main detection script
│ │ ├── fetch_book_info.py # Book metadata fetching
│ │ └── fetch_database.py # Database operations
│ └── src/
│ └── server.js # Backend server
├── frontend/
│ └── public/
│ ├── index.html # Web interface
│ └── index.js # Frontend logic
└── models/ # AI model files
```


## Setup Instructions

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Install Node.js dependencies:
   ```bash
   cd backend
   npm install
   ```
4. Set up required API keys and credentials
5. Place model files in the appropriate directories

## Usage

### Command Line
``` bash
python backend/python-scripts/detect.py <path_to_image>
```


### Web Interface
1. Start the backend server:
   ```bash
   cd backend
   npm start
   ```
2. Open `frontend/public/index.html` in a web browser
3. Upload an image containing book spines
4. View the detected books and extracted metadata

## Output

The system generates:
- Detected book metadata (title, author)
- Enhanced images
- Cropped individual book spine images
- Annotated original image showing detections
- Cached results for faster subsequent processing

## Caching System

The system implements multi-level caching to improve performance and reduce API costs:
- OCR results cache
- Gemini API response cache
- Full process results cache

Cache files are stored in the output directory structure:
```
output/
└── image_name/
├── crops/ # Cropped book spine images
├── ocr_cache/ # OCR results
├── gemini_cache/ # AI refinement results
└── process_cache/ # Full process results
```

## License

[Add your license information here]

## Contributors

[Add contributor information here]

