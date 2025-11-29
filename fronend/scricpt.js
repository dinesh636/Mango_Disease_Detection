const form = document.querySelector("form");
const result = document.querySelector("#result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.querySelector("#image");
  const file = fileInput.files[0];

  if (!file) {
    result.textContent = "Please upload an image first!";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  result.textContent = "Analyzing...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      result.textContent = `❌ Error: ${data.error}`;
    } else {
      result.innerHTML = `✅ <b>Prediction:</b> ${data.class}<br>📊 Confidence: ${data.confidence}`;
    }
  } catch (err) {
    result.textContent = "⚠️ Failed to connect to backend!";
    console.error(err);
  }
});
