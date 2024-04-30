const { spawn } = require('child_process');
const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 5000;

// Parse JSON bodies
app.use(bodyParser.json());

// Serve static files from the React app
app.use(express.static(path.join(__dirname, './build')));

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, './build', 'index.html'));
  });

// Endpoint for predicting used car prices
app.post('/api/predict_price', (req, res) => {
    // Get features from request body
    const features = req.body.features;
    console.log(req.body);
    console.log(features);
    
    // Spawn a Python child process to run the prediction script
    const pythonProcess = spawn('python', ['predict.py', JSON.stringify(features)]);
    
    // Collect data from the Python process
    let data = '';
    pythonProcess.stdout.on('data', (chunk) => {
        data += chunk.toString();
    });
    console.log('data,' , data);

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });
    
    pythonProcess.on('error', (error) => {
        console.error(`Failed to start subprocess: ${error}`);
    });    
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
        console.log(code);
        if (code === 0) {

            // Send the predicted price as a response
            res.json({ price: parseInt(data) });
        } else {
            res.status(500).send('Failed to predict price');
        }
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
