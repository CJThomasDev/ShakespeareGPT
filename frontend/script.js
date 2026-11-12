// Get our webpage elements
const temperatureSlider = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const promptSelect = document.getElementById("prompt");
const lengthSelect = document.getElementById("length");
const generateButton = document.getElementById("generate-btn");
const output = document.getElementById("output");

// Our local FastAPI endpoint
const API_URL = "http://127.0.0.1:8000/generate";


// Update the creativity value when the slider moves
temperatureSlider.addEventListener("input", function () {
    temperatureValue.textContent = temperatureSlider.value;
});


// Generate dialogue when the button is clicked
generateButton.addEventListener("click", async function () {

    // Get the user's settings
    const prompt = promptSelect.value;
    const temperature = parseFloat(temperatureSlider.value);
    const newChars = parseInt(lengthSelect.value, 10);

    // Show that the model is generating
    generateButton.disabled = true;
    generateButton.textContent = "Generating...";

    output.textContent = "Your scene is being written...";
    output.classList.remove("has-content");

    try {

        // Send the settings to our FastAPI backend
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt: prompt,
                temperature: temperature,
                new_chars: newChars
            })
        });

        // Check whether the request worked
        if (!response.ok) {
            throw new Error("The API returned an error: " + response.status);
        }

        // Read the generated dialogue
        const data = await response.json();

        // Display the result
        output.textContent = data.generated_text;
        output.classList.add("has-content");

    } catch (error) {

        // Display an error if the API cannot be reached
        output.textContent = "Could not generate dialogue. Make sure the API server is running.";
        console.error(error);

    } finally {

        // Reset the button
        generateButton.disabled = false;
        generateButton.textContent = "Generate Dialogue";
    }
});