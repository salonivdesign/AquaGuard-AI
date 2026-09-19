const form = document.getElementById("waterForm");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const data = {
        ph: Number(document.getElementById("ph").value),
        Hardness: Number(document.getElementById("Hardness").value),
        Solids: Number(document.getElementById("Solids").value),
        Chloramines: Number(document.getElementById("Chloramines").value),
        Sulfate: Number(document.getElementById("Sulfate").value),
        Conductivity: Number(document.getElementById("Conductivity").value),
        Organic_carbon: Number(document.getElementById("Organic_carbon").value),
        Trihalomethanes: Number(document.getElementById("Trihalomethanes").value),
        Turbidity: Number(document.getElementById("Turbidity").value)
    };
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("overallRisk").textContent =
            result.overall_risk;

        document.getElementById("potabilityRisk").textContent =
            result.potability_risk;

        document.getElementById("anomalyStatus").textContent =
            result.anomaly_status;

        document.getElementById("probability").textContent =
            (result.potability_probability * 100).toFixed(2) + "%";

        document.getElementById("recommendation").textContent =
            result.recommendation;

    } catch (error) {
        console.error(error);

        document.getElementById("recommendation").textContent =
            "Unable to connect to the AquaGuard AI backend.";
    }
});

async function loadHistory() {
    try {
        const response = await fetch("http://127.0.0.1:8000/history");

        const history = await response.json();

        const historyBody = document.getElementById("historyBody");

        if (history.length === 0) {
            historyBody.innerHTML = `
                <tr>
                    <td colspan="5">No analysis history available.</td>
                </tr>
            `;
            return;
        }

        historyBody.innerHTML = "";

        history.forEach(item => {
            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${item.created_at}</td>
                <td>${item.overall_risk}</td>
                <td>${item.potability_risk}</td>
                <td>${item.anomaly_status}</td>
                <td>${(item.potability_probability * 100).toFixed(2)}%</td>
            `;

            historyBody.appendChild(row);
        });

    } catch (error) {
        console.error(error);

        document.getElementById("historyBody").innerHTML = `
            <tr>
                <td colspan="5">Unable to load analysis history.</td>
            </tr>
        `;
    }
}

loadHistory();