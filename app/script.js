var codeInput = document.getElementById('codeInput');
var predictions = document.getElementById('predictions');

var typingTimer;
var doneTypingInterval = document.getElementById('typingInterval').value * 1000;

document.getElementById('typingInterval').oninput = function () {
    doneTypingInterval = this.value * 1000;
    document.getElementById('typingIntervalValue').innerHTML = this.value;
}

codeInput.addEventListener('input', function () {
    clearTimeout(typingTimer);
    typingTimer = setTimeout(function () {
        var code = codeInput.value;
        var modelType = document.querySelector('input[name="modelType"]:checked').value;

        predictions.innerHTML = `
        <div class="spinner-border text-light" role="status" style="margin-top: 1%">
            <span class="sr-only">Loading...</span>
        </div>`;

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'code': code, 'model_type': modelType })
        })
            .then(response => response.json())
            .then(data => {
                var formattedPredictions = data.predictions.map(prediction => `<span class="token">${prediction}</span>`).join(' ');
                predictions.innerHTML = `<pre class="language-python line-numbers">${formattedPredictions}</pre>`;

                Prism.highlightAll();
            })
            .catch(error => {
                predictions.innerHTML = `<p class="text-danger">Error: ${error}</p>`;
            });
    }, doneTypingInterval);
});

predictions.addEventListener('click', function (event) {
    if (event.target && event.target.matches('span.token')) {
        codeInput.value += event.target.innerText + ' ';
    }
});
