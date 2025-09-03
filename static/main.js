// JavaScript for Medical Image Analysis System
document.addEventListener('DOMContentLoaded', function () {
	const fileInput = document.getElementById('fileInput');
	const fileStatus = document.getElementById('fileStatus');
	const resultsSection = document.getElementById('resultsSection');
	const resultsContent = document.getElementById('resultsContent');

	fileInput.addEventListener('change', function () {
		if (fileInput.files.length > 0) {
			fileStatus.textContent = `Selected: ${fileInput.files[0].name}`;
			uploadFile(fileInput.files[0]);
		}
	});

	function uploadFile(file) {
		fileStatus.textContent = 'Uploading and analyzing...';
		resultsSection.style.display = 'none';
		resultsContent.innerHTML = '';

		const formData = new FormData();
		formData.append('file', file);

		fetch('/analyze', {
			method: 'POST',
			body: formData
		})
		.then(response => response.json())
		.then(data => {
			if (data.error) {
				fileStatus.textContent = 'Error: ' + data.error;
				return;
			}
			fileStatus.textContent = 'Analysis complete!';
			showResults(data);
		})
		.catch(err => {
			fileStatus.textContent = 'Upload failed.';
		});
	}

	function showResults(data) {
		resultsSection.style.display = 'block';
		let html = '';
		if (data.segmentation_results) {
			html += `<p><strong>Cells Detected:</strong> ${data.segmentation_results.total_cells_detected}</p>`;
			html += `<p><strong>Segmentation Method:</strong> ${data.segmentation_results.segmentation_method}</p>`;
		}
		if (data.visualizations) {
			if (data.visualizations.original) {
				html += `<div><strong>Original Image:</strong><br><img src="${data.visualizations.original}" style="max-width:400px;"></div>`;
			}
			if (data.visualizations.segmented) {
				html += `<div><strong>Segmented Image:</strong><br><img src="${data.visualizations.segmented}" style="max-width:400px;"></div>`;
			}
			if (data.visualizations.overlay) {
				html += `<div><strong>Overlay:</strong><br><img src="${data.visualizations.overlay}" style="max-width:400px;"></div>`;
			}
		}
		resultsContent.innerHTML = html;
	}
});
