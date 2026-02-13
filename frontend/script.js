async function analyzeSentiment() {
    const text = document.getElementById("userText").value;
    const resultDiv = document.getElementById("result");

    if (!text) {
        resultDiv.innerHTML = "⚠ Please enter some text!";
        resultDiv.style.color = "orange";
        return;
    }

    resultDiv.innerHTML = "⏳ Analyzing...";
    resultDiv.style.color = "black";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.predicted_sentiment === "positive") {
            resultDiv.innerHTML = "😊 Positive Sentiment";
            resultDiv.style.color = "green";
        } else {
            resultDiv.innerHTML = "😡 Negative Sentiment";
            resultDiv.style.color = "red";
        }

    } catch (error) {
        resultDiv.innerHTML = "❌ Error connecting to server";
        resultDiv.style.color = "red";
    }
}
