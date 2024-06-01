const express = require('express');
const request = require('request');

const app = express();
const port = 3000; // Choose any available port

// Enable CORS (Cross-Origin Resource Sharing)
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});

app.get('/getNearbyDoctors', (req, res) => {
    const apiKey = 'AIzaSyD6d8HkcFafZCsES0o5aKg0Eyt_W4jWGn8'; // Replace with your Google Places API key
    const location = req.query.location;
    const radius = req.query.radius || 5000; // Default radius of 5000 meters

    const apiUrl = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${location}&radius=${radius}&type=doctor&key=${apiKey}`;

    request(apiUrl, (error, response, body) => {
        if (!error && response.statusCode === 200) {
            res.send(body);
        } else {
            res.status(500).send('Error');
        }
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
