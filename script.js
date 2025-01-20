// static/script.js

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const loading = document.getElementById('loading');
    const resultContent = document.getElementById('result-content');
    const downloadBtn = document.getElementById('download-btn');

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Clear previous results
        resultContent.innerHTML = '';
        downloadBtn.style.display = 'none';

        const file = imageInput.files[0];
        if (!file) {
            alert('Please select an image to upload.');
            return;
        }

        // Preview the image
        const reader = new FileReader();
        reader.onload = function(event) {
            imagePreview.src = event.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(file);

        // Show loading animation
        loading.style.display = 'flex';

        // Prepare form data
        const formData = new FormData();
        formData.append('image', file);

        // Send the image to the Flask backend
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading animation
            loading.style.display = 'none';

            if (data.error) {
                resultContent.innerHTML = `<span class="error">${data.error}</span>`;
                return;
            }

            // Display the JSON result
            const formattedResult = JSON.stringify(data, null, 4);
            resultContent.textContent = formattedResult;

            // Show download button
            downloadBtn.style.display = 'inline-block';
            downloadBtn.onclick = function() {
                downloadResults(data);
            };
        })
        .catch(error => {
            loading.style.display = 'none';
            resultContent.innerHTML = `<span class="error">An error occurred: ${error}</span>`;
            console.error('Error:', error);
        });
    });

    function downloadResults(data) {
        fetch('/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to download results.');
            }
            return response.blob();
        })
        .then(blob => {
            // Create a link to download the blob
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'result.txt';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            alert('Error downloading the file.');
            console.error('Download Error:', error);
        });
    }
});
