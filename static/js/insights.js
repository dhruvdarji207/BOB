/**
 * insights.js — Chart.js visualizations for the Model Insights page.
 * Depends on: Chart.js (CDN), and window.METRICS set by the Jinja template.
 */

(function () {
  if (typeof METRICS === "undefined" || !METRICS) return;

  const ACCENT       = "#3b82f6";
  const ACCENT2      = "#60a5fa";
  const PURPLE       = "#8b5cf6";
  const SUCCESS      = "#22c55e";
  const WARNING      = "#f59e0b";
  const BORDER_COLOR = "#2d3748";
  const TEXT_COLOR   = "#e6edf3";
  const MUTED_COLOR  = "#8b949e";

  // Shared chart defaults
  Chart.defaults.color = MUTED_COLOR;
  Chart.defaults.borderColor = BORDER_COLOR;
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size   = 12;

  // ── 1. R² Bar chart ───────────────────────────────────────────
  const r2Ctx = document.getElementById("r2Chart");
  if (r2Ctx && METRICS.all_models) {
    const modelNames = Object.keys(METRICS.all_models);
    const r2Values   = modelNames.map(n => METRICS.all_models[n].r2);
    const bestModel  = METRICS.best_model;

    const barColors = modelNames.map(n =>
      n === bestModel ? ACCENT : "rgba(59,130,246,0.35)"
    );

    new Chart(r2Ctx, {
      type: "bar",
      data: {
        labels: modelNames.map(shortenName),
        datasets: [{
          label: "R² Score",
          data: r2Values,
          backgroundColor: barColors,
          borderColor: barColors,
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` R² = ${ctx.parsed.y.toFixed(4)}`,
            },
          },
        },
        scales: {
          y: {
            min: 0,
            max: 1,
            grid: { color: BORDER_COLOR },
            ticks: { color: MUTED_COLOR },
          },
          x: {
            grid: { display: false },
            ticks: { color: MUTED_COLOR, maxRotation: 30 },
          },
        },
      },
    });
  }

  // ── 2. Price distribution bar chart ───────────────────────────
  const distCtx = document.getElementById("priceDistChart");
  if (distCtx && METRICS.price_distribution) {
    const dist = METRICS.price_distribution;
    new Chart(distCtx, {
      type: "bar",
      data: {
        labels: dist.map(d => d.range),
        datasets: [{
          label: "Number of Cars",
          data: dist.map(d => d.count),
          backgroundColor: "rgba(139,92,246,0.5)",
          borderColor: PURPLE,
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.y} cars`,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: BORDER_COLOR },
            ticks: { color: MUTED_COLOR, stepSize: 10 },
          },
          x: {
            grid: { display: false },
            ticks: {
              color: MUTED_COLOR,
              maxRotation: 45,
              font: { size: 9 },
            },
          },
        },
      },
    });
  }

  // ── 3. Feature importance horizontal bar chart ─────────────────
  const fiCtx = document.getElementById("featureChart");
  if (fiCtx && METRICS.feature_importance && METRICS.feature_importance.length > 0) {
    const top15 = METRICS.feature_importance.slice(0, 15);
    const labels      = top15.map(d => d.feature).reverse();
    const importances = top15.map(d => d.importance).reverse();

    // Colour gradient by rank
    const bgColors = importances.map((_, i) => {
      const alpha = 0.35 + 0.65 * (i / (importances.length - 1));
      return `rgba(59,130,246,${alpha.toFixed(2)})`;
    });

    new Chart(fiCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Importance",
          data: importances,
          backgroundColor: bgColors,
          borderColor: bgColors.map(c => c.replace(/,[0-9.]+\)/, ",0.9)")),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.x.toFixed(4)}`,
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: BORDER_COLOR },
            ticks: { color: MUTED_COLOR },
          },
          y: {
            grid: { display: false },
            ticks: { color: TEXT_COLOR, font: { size: 11 } },
          },
        },
      },
    });
  }

  // ── 4. Actual vs Predicted scatter chart ───────────────────────
  const scCtx = document.getElementById("scatterChart");
  if (scCtx && METRICS.actual_vs_predicted && METRICS.actual_vs_predicted.length > 0) {
    const avp = METRICS.actual_vs_predicted;
    const scatterData = avp.map(d => ({ x: d.actual, y: d.predicted }));

    // Perfect prediction line (min to max)
    const allActual = avp.map(d => d.actual);
    const minVal    = Math.min(...allActual);
    const maxVal    = Math.max(...allActual);
    const idealLine = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }];

    new Chart(scCtx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Predictions",
            data: scatterData,
            backgroundColor: "rgba(59,130,246,0.5)",
            borderColor: ACCENT,
            pointRadius: 5,
            pointHoverRadius: 7,
          },
          {
            label: "Perfect Fit (y=x)",
            data: idealLine,
            type: "line",
            borderColor: SUCCESS,
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: { color: MUTED_COLOR, boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: ctx => {
                if (ctx.datasetIndex === 0) {
                  return ` Actual: $${formatK(ctx.parsed.x)}  Predicted: $${formatK(ctx.parsed.y)}`;
                }
                return ` Perfect fit`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Actual Price ($)", color: MUTED_COLOR },
            grid: { color: BORDER_COLOR },
            ticks: { color: MUTED_COLOR, callback: v => "$" + formatK(v) },
          },
          y: {
            title: { display: true, text: "Predicted Price ($)", color: MUTED_COLOR },
            grid: { color: BORDER_COLOR },
            ticks: { color: MUTED_COLOR, callback: v => "$" + formatK(v) },
          },
        },
      },
    });
  }

  /* ── Utilities ── */
  function shortenName(name) {
    const map = {
      "Linear Regression":          "Linear",
      "Ridge Regression":           "Ridge",
      "Random Forest":              "RF",
      "Gradient Boosting":          "GBR",
      "Extra Trees":                "ET",
      "HistGradientBoosting":       "HistGB",
      "HistGradientBoostingRegressor": "HistGB",
    };
    return map[name] || name;
  }

  function formatK(v) {
    return v >= 1000 ? (v / 1000).toFixed(0) + "k" : v.toFixed(0);
  }
})();
