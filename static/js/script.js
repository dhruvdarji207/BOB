/**
 * script.js — Global frontend logic for CarPriceAI
 * Handles: navigation toggle, prediction form submit, result display
 */

/* ── Navigation toggle (mobile) ─────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const links  = document.getElementById("navLinks");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      links.classList.toggle("open");
    });
    // Close when a link is clicked
    links.querySelectorAll("a").forEach(a => {
      a.addEventListener("click", () => links.classList.remove("open"));
    });
  }

  // Wire up prediction form if on predict page
  const form = document.getElementById("predictForm");
  if (form) initPredictForm(form);
});


/* ── Prediction form ────────────────────────────────────────────── */
function initPredictForm(form) {
  const predictBtn       = document.getElementById("predictBtn");
  const btnText          = document.querySelector(".btn-text");
  const btnSpinner       = document.getElementById("btnSpinner");
  const errorBanner      = document.getElementById("errorBanner");
  const resultCard       = document.getElementById("resultCard");
  const resultPlaceholder = document.getElementById("resultPlaceholder");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    const data = collectFormData(form);
    const validationError = validateClient(data);
    if (validationError) {
      showError(validationError);
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (!result.success) {
        showError(result.error || "Prediction failed. Please check your inputs.");
        return;
      }

      displayResult(result, data);

    } catch (err) {
      showError("Network error. Is the server running?");
    } finally {
      setLoading(false);
    }
  });

  // Reset button
  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      clearErrors();
      resultCard.classList.add("hidden");
      resultPlaceholder.classList.remove("hidden");
    });
  }

  /* ── Helpers ─── */

  function collectFormData(form) {
    const fd = new FormData(form);
    const data = {};
    for (const [key, val] of fd.entries()) {
      // Numeric fields
      const numFields = [
        "symboling","wheelbase","carlength","carwidth","carheight",
        "curbweight","enginesize","boreratio","stroke","compressionratio",
        "horsepower","peakrpm","citympg","highwaympg",
      ];
      if (numFields.includes(key)) {
        data[key] = val === "" ? "" : parseFloat(val);
      } else {
        data[key] = val;
      }
    }
    return data;
  }

  function validateClient(data) {
    const required = [
      "manufacturer","fueltype","aspiration","doornumber","carbody",
      "drivewheel","enginelocation","symboling","enginetype","cylindernumber",
      "fuelsystem","enginesize","horsepower","peakrpm","boreratio","stroke",
      "compressionratio","wheelbase","carlength","carwidth","carheight",
      "curbweight","citympg","highwaympg",
    ];
    const missing = required.filter(f => data[f] === "" || data[f] === undefined || data[f] === null);
    if (missing.length > 0) {
      // Highlight invalid fields
      missing.forEach(f => {
        const el = document.getElementById(f);
        if (el) el.classList.add("invalid");
      });
      return `Please fill in all required fields: ${missing.slice(0,4).join(", ")}${missing.length > 4 ? ` and ${missing.length-4} more` : ""}.`;
    }

    const numFields = ["enginesize","horsepower","peakrpm","wheelbase","carlength","curbweight","citympg","highwaympg"];
    for (const f of numFields) {
      if (isNaN(data[f]) || data[f] <= 0) {
        const el = document.getElementById(f);
        if (el) el.classList.add("invalid");
        return `"${f}" must be a positive number.`;
      }
    }
    return null;
  }

  function setLoading(loading) {
    predictBtn.disabled = loading;
    if (loading) {
      btnText.classList.add("hidden");
      btnSpinner.classList.remove("hidden");
    } else {
      btnText.classList.remove("hidden");
      btnSpinner.classList.add("hidden");
    }
  }

  function showError(msg) {
    errorBanner.textContent = "⚠ " + msg;
    errorBanner.classList.remove("hidden");
    errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function clearErrors() {
    errorBanner.textContent = "";
    errorBanner.classList.add("hidden");
    document.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
  }

  function displayResult(result, inputData) {
    resultPlaceholder.classList.add("hidden");
    resultCard.classList.remove("hidden");

    // Format price
    const price = result.predicted_price;
    document.getElementById("resultPrice").textContent = formatUSD(price);
    document.getElementById("modelUsed").textContent   = result.model_used || "ML Model";
    document.getElementById("modelR2").textContent     =
      result.model_r2 != null ? result.model_r2.toFixed(4) : "—";

    // Summary
    const summary = document.getElementById("resultSummary");
    summary.textContent =
      `Input summary: ${inputData.manufacturer || "—"} · ${inputData.carbody || "—"} · ` +
      `${inputData.fueltype || "—"} · ${inputData.enginesize || "—"}cc · ` +
      `${inputData.horsepower || "—"}hp · ${inputData.citympg || "—"}/${inputData.highwaympg || "—"} mpg`;

    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

/* ── Formatters ─────────────────────────────────────────────────── */
function formatUSD(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}
