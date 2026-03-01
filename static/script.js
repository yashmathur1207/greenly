// Load plants from data.json dynamically
async function loadPlantData() {
    const res = await fetch("static/data.json");
    const data = await res.json();
    return data;
}

let plantDataGlobal = {};

async function init() {
    plantDataGlobal = await loadPlantData();
}

init();

function selectCategory(category) {
    document.getElementById("selection-screen").classList.add("hidden");
    document.getElementById("category-screen").classList.remove("hidden");
    document.getElementById("back-link").style.display = "block";

    const grid = document.getElementById("plant-grid");
    grid.innerHTML = "";

    plantDataGlobal[category].forEach(plant => {
        const card = document.createElement("div");
        card.className = "bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-xl transition";
        card.innerHTML = `
            <img src="${plant.image}" class="w-full h-40 object-cover">
            <div class="p-4">
                <h3 class="font-bold text-lg">${plant.name}</h3>
                <p class="text-sm text-gray-600">${plant.short_description}</p>
                <button onclick="predictPlant('${plant.name}')" 
                    class="mt-2 bg-primary text-white px-3 py-1 rounded hover:bg-dark transition">Predict Disease</button>
            </div>
        `;
        card.querySelector("button").addEventListener("click", e => {
            e.stopPropagation(); // Prevent opening modal
        });
        card.onclick = () => showPlantDetail(plant);
        grid.appendChild(card);
    });
}

function showPlantDetail(plant) {
    const modal = document.getElementById("plant-modal");
    const content = document.getElementById("modal-content");
    content.innerHTML = `
        <h2 class="text-xl font-bold mb-2">${plant.name}</h2>
        <img src="${plant.image}" class="w-full h-48 object-cover rounded mb-4">
        <p>${plant.description}</p>
        <p class="mt-2 font-semibold">Care:</p>
        <p>${plant.care}</p>
    `;
    modal.classList.remove("hidden");
}

function closeModal() {
    document.getElementById("plant-modal").classList.add("hidden");
}

function goBack() {
    document.getElementById("category-screen").classList.add("hidden");
    document.getElementById("selection-screen").classList.remove("hidden");
    document.getElementById("back-link").style.display = "none";
}

// ---------------- Prediction ----------------
async function predictPlant(plantName) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
        const file = input.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);

        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        alert(`Prediction for ${plantName}:\n\nResult: ${data.prediction}\nSolution: ${data.solution}`);
    };
    input.click();
}