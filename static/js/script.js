document.addEventListener('DOMContentLoaded', () => {
    const videoFeed = document.getElementById('videoFeed');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');
    const terminal = document.getElementById('terminalLog');

    // The endpoint defined in your Flask app.py
    // Assuming standard: @app.route('/video_feed')
    const VIDEO_URL = "/video_feed"; 

    function log(message, type = 'system') {
        const p = document.createElement('p');
        p.className = `log-entry ${type}`;
        
        // Add timestamp
        const time = new Date().toLocaleTimeString('en-US', {hour12: false});
        p.innerText = `[${time}] > ${message}`;
        
        terminal.appendChild(p);
        terminal.scrollTop = terminal.scrollHeight; // Auto scroll
    }

    startBtn.addEventListener('click', () => {
        log("Initializing camera stream...", "system");
        
        // Visual updates
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusDot.classList.add('active');
        statusText.innerText = "SYSTEM ACTIVE";
        statusText.style.color = "#00ff00";
        videoFeed.classList.add('active');

        // Set the source to the Flask route
        // Adding a timestamp to prevent browser caching issues
        videoFeed.src = `${VIDEO_URL}?t=${new Date().getTime()}`;
        
        log("Connection established.", "success");
        log("Model: Random Forest (Active)", "system");
        
        // Simulation of processing
        simulateProcessing();
    });

    stopBtn.addEventListener('click', () => {
        log("Terminating signal...", "error");
        
        // Visual updates
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusDot.classList.remove('active');
        statusText.innerText = "SYSTEM STANDBY";
        statusText.style.color = "#888";
        videoFeed.classList.remove('active');

        // Kill the source
        videoFeed.src = "";
        
        log("Feed disconnected.", "system");
    });

    // Just for visual flair - simulates latency numbers updating
    function simulateProcessing() {
        if(!startBtn.disabled) return;
        
        const latency = document.getElementById('latencyVal');
        const confidence = document.getElementById('confidenceVal');
        
        // Random fluctuation
        const lat = Math.floor(Math.random() * (45 - 20 + 1) + 20);
        const conf = Math.floor(Math.random() * (99 - 85 + 1) + 85);
        
        latency.innerText = `${lat}ms`;
        confidence.innerText = `${conf}%`;
        
        setTimeout(simulateProcessing, 2000);
    }
});