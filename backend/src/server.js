const express = require("express")
const multer = require("multer")
const cors = require("cors")
const fs = require("fs")

const { spawn } = require("child_process")
const path = require("path")

// Initialize Express app
const app = express()
app.use(cors({
  origin: 'http://127.0.0.1:5501', // Add your Live Server URL here
  credentials: true
}))
const PORT = process.env.PORT || 3000
app.use(express.static("../frontend/public"))
app.use('/output', express.static(path.join(__dirname, '../python-scripts/output')))

// Set up storage for uploaded images
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.resolve(__dirname, "uploads")) // Save files to 'uploads' folder
  },
  filename: (req, file, cb) => {
    // just use the orignial name
    cb(null, file.originalname)
    // cb(null, Date.now() + path.extname(file.originalname)) // Use timestamp as filename
  },
})

const upload = multer({ storage: storage })

// POST route to handle the image upload
app.post("/upload", upload.single("image"), (req, res) => {
  const filePath = req.file.path
  const pythonScriptPath = path.resolve(__dirname, "../python-scripts/main.py")
  const CONDA_PATH = 'C:\\Users\\p9380\\miniconda3\\envs\\cv\\python.exe'
  const pythonScriptDir = path.dirname(pythonScriptPath)
  console.log("pythonScriptDir: ", pythonScriptDir)
  const pythonProcess = spawn(
    CONDA_PATH,
    [pythonScriptPath, filePath],
    {
      env: { ...process.env },
      cwd: pythonScriptDir
    }
  );

  let outputData = "";

  pythonProcess.stdout.on("data", (data) => {
    outputData += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    try {
      console.log("Raw Python output:", outputData);
      
      const match = outputData.match(/BOOK_DETECTION_RESULT:([\s\S]+)/);
      
      if (match) {
        const jsonResult = JSON.parse(match[1].trim());
        const imageName = path.basename(filePath);
        const annotatedImagePath = path.join(
          __dirname, 
          '../python-scripts/output', 
          path.parse(imageName).name,
          `${path.parse(imageName).name}_annotated.png`
        );
        
        // Read the annotated image and convert to base64
        let annotatedImageBase64 = '';
        if (fs.existsSync(annotatedImagePath)) {
          const imageBuffer = fs.readFileSync(annotatedImagePath);
          annotatedImageBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
        }
        
        res.json({ 
          result: jsonResult,
          annotatedImage: annotatedImageBase64,
          success: true
        });
      } else {
        res.status(500).json({ 
          error: "No valid result found",
          raw: outputData,  // Include raw output in error response
          success: false
        });
      }
    } catch (e) {
      console.error("Error processing result:", e);
      res.status(500).json({ 
        error: "Failed to parse result",
        raw: outputData,  // Include raw output in error response
        success: false
      });
    }
  });
})

// POST route to handle the image upload
app.post('/detect', upload.single('image'), async (req, res) => {
  try {
    const imagePath = req.file.path;
    const pythonProcess = spawn('python', [pythonScriptPath, imagePath]);
    
    let result = '';
    
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });
    
    pythonProcess.on('close', async (code) => {
      try {
        // Extract the JSON part from the Python output
        const jsonMatch = result.match(/BOOK_DETECTION_RESULT:(.+)/s);
        if (!jsonMatch) {
          throw new Error('Invalid output format from Python script');
        }
        
        const bookData = JSON.parse(jsonMatch[1]);
        
        // Get the annotated image if it exists
        const annotatedImagePath = imagePath + '_annotated.jpg';
        let annotatedImageBase64 = null;
        
        if (fs.existsSync(annotatedImagePath)) {
          annotatedImageBase64 = 'data:image/jpeg;base64,' + 
            fs.readFileSync(annotatedImagePath, {encoding: 'base64'});
        }
        
        // Clean up temporary files
        fs.unlinkSync(imagePath);
        if (fs.existsSync(annotatedImagePath)) {
          fs.unlinkSync(annotatedImagePath);
        }
        
        res.json({
          success: true,
          books: bookData,
          annotatedImage: annotatedImageBase64
        });
        
      } catch (error) {
        console.error('Error processing Python output:', error);
        res.status(500).json({
          success: false,
          error: 'Error processing detection results'
        });
      }
    });
    
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({
      success: false,
      error: 'Server error processing request'
    });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})
