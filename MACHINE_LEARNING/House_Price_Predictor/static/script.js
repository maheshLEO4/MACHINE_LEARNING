document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const title = document.getElementById('title').value;
    const location = document.getElementById('location').value;
    const area = parseFloat(document.getElementById('area').value);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('predictBtn').textContent = '⏳ Processing...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                location: location,
                area: area
            })
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        if (response.ok) {
            // Show success result
            document.getElementById('result').innerHTML = `
                <div class="price-display">₹${data['predicted_price(L)']} Lakhs</div>
                <div class="details">
                    <div class="detail-row">
                        <span><strong>Property Type:</strong></span>
                        <span>${data.title}</span>
                    </div>
                    <div class="detail-row">
                        <span><strong>Location:</strong></span>
                        <span>${data.location}</span>
                    </div>
                    <div class="detail-row">
                        <span><strong>Area:</strong></span>
                        <span>${data.area_insqft} sq ft</span>
                    </div>
                    <div class="detail-row">
                        <span><strong>Price per sq ft:</strong></span>
                        <span>₹${Math.round((data['predicted_price(L)'] * 100000) / data.area_insqft).toLocaleString()}</span>
                    </div>
                </div>
            `;
            document.getElementById('result').className = 'result success';
            document.getElementById('result').style.display = 'block';
        } else {
            // Show error
            document.getElementById('result').innerHTML = `
                <strong>❌ Error:</strong> ${data.error}
            `;
            document.getElementById('result').className = 'result error';
            document.getElementById('result').style.display = 'block';
        }
    } catch (error) {
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        // Show error
        document.getElementById('result').innerHTML = `
            <strong>❌ Connection Error:</strong> ${error.message}
        `;
        document.getElementById('result').className = 'result error';
        document.getElementById('result').style.display = 'block';
    }
    
    // Reset button
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('predictBtn').textContent = '🔮 Predict Price';
});

// Add some visual feedback for form inputs
const inputs = document.querySelectorAll('input, select');
inputs.forEach(input => {
    input.addEventListener('focus', function() {
        this.style.transform = 'scale(1.02)';
    });
    
    input.addEventListener('blur', function() {
        this.style.transform = 'scale(1)';
    });
});