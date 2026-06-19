/* =========================================================
   Healthcare SaaS Dashboard — static/js/dashboard.js
   Dark mode toggle · Loading spinner · Plotly gauge & SHAP
   charts · Form validation
   ========================================================= */

(function (window, document) {
  "use strict";

  /* =========================================================
     1. DARK MODE TOGGLE
     - Persists choice in localStorage under "ews-theme"
     - Falls back to OS preference on first visit
     - Toggle any element with [data-theme-toggle]
     ========================================================= */
  const THEME_KEY = "ews-theme";

  function getStoredTheme() {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch (e) {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (e) {
      /* localStorage unavailable (private mode, etc.) — ignore */
    }
  }

  function systemPrefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      btn.setAttribute("title", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    });
    document.dispatchEvent(new CustomEvent("ews:themechange", { detail: { theme } }));
  }

  function initTheme() {
    const stored = getStoredTheme();
    const theme = stored || (systemPrefersDark() ? "dark" : "light");
    applyTheme(theme);

    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        storeTheme(next);
        // Re-theme any rendered Plotly charts so they don't look mismatched
        refreshChartsForTheme(next);
      });
    });

    // Keep in sync if the OS theme changes and the user hasn't explicitly chosen one
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
        if (!getStoredTheme()) applyTheme(e.matches ? "dark" : "light");
      });
    }
  }

  /* =========================================================
     2. LOADING SPINNER
     - showSpinner(target) swaps a button/container's content
       for a spinner and disables it
     - hideSpinner(target) restores the original content
     ========================================================= */
  const SPINNER_HTML = '<span class="spinner" role="status" aria-label="Loading"></span>';
  const spinnerState = new WeakMap();

  function showSpinner(target, label) {
    const el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) return;

    if (!spinnerState.has(el)) {
      spinnerState.set(el, { html: el.innerHTML, disabled: el.disabled });
    }

    if (el.tagName === "BUTTON" || el.tagName === "A") {
      el.disabled = true;
      el.setAttribute("aria-busy", "true");
      el.innerHTML = SPINNER_HTML + (label ? `<span>${label}</span>` : "");
    } else {
      el.setAttribute("aria-busy", "true");
      el.innerHTML = `<div class="flex items-center gap-12">${SPINNER_HTML}<span>${label || "Loading…"}</span></div>`;
    }
  }

  function hideSpinner(target) {
    const el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) return;
    const saved = spinnerState.get(el);
    if (saved) {
      el.innerHTML = saved.html;
      el.disabled = saved.disabled;
      spinnerState.delete(el);
    }
    el.removeAttribute("aria-busy");
  }

  /* Auto-spin any form carrying [data-spinner-on-submit] */
  function initAutoSpinners() {
    document.querySelectorAll("form[data-spinner-on-submit]").forEach((form) => {
      form.addEventListener("submit", (e) => {
        if (form.dataset.validated === "false") return; // validation failed, don't spin
        const submitBtn = form.querySelector('[type="submit"]');
        if (submitBtn) showSpinner(submitBtn, submitBtn.dataset.loadingLabel || "Processing…");
      });
    });
  }

  /* =========================================================
     3. PLOTLY GAUGE CHART
     ========================================================= */
  const chartRegistry = new Map(); // elementId -> { type, render fn args } for theme refresh

  function plotlyColors() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    return {
      ink: dark ? "#eef3fc" : "#0f1c2e",
      track: dark ? "rgba(255,255,255,0.08)" : "#e2e9f7",
      bar: dark ? "#60a5fa" : "#152826",
      good: dark ? "#1f3d33" : "#bfe3cf",
      warn: dark ? "#4a3a14" : "#f1d99a",
      danger: dark ? "#4a2222" : "#eab3ad",
      thresholdLine: dark ? "#fb7185" : "#a93b34",
      paper: "rgba(0,0,0,0)",
    };
  }

  /**
   * Render a Plotly probability gauge.
   * @param {string} elementId - container element id
   * @param {number} value - probability 0..1
   * @param {number} threshold - alert threshold 0..1
   * @param {number} [maxDisplay=0.5] - gauge axis cap (0..1)
   */
  function renderGaugeChart(elementId, value, threshold, maxDisplay) {
    if (typeof Plotly === "undefined") {
      console.warn("Plotly.js is not loaded; gauge chart skipped.");
      return;
    }
    const el = document.getElementById(elementId);
    if (!el) return;

    const cap = (maxDisplay || 0.5) * 100;
    const c = plotlyColors();

    const data = [
      {
        type: "indicator",
        mode: "gauge+number",
        value: value * 100,
        number: { suffix: "%", font: { size: 34, color: c.ink } },
        gauge: {
          axis: { range: [0, cap], ticksuffix: "%", tickfont: { color: c.ink } },
          bar: { color: c.bar },
          bgcolor: c.paper,
          borderwidth: 0,
          steps: [
            { range: [0, Math.min(6, cap)], color: c.good },
            { range: [Math.min(6, cap), Math.min(threshold * 100, cap)], color: c.warn },
            { range: [Math.min(threshold * 100, cap), cap], color: c.danger },
          ],
          threshold: {
            line: { color: c.thresholdLine, width: 3 },
            thickness: 0.85,
            value: Math.min(threshold * 100, cap),
          },
        },
      },
    ];

    const layout = {
      margin: { t: 24, b: 8, l: 30, r: 30 },
      paper_bgcolor: c.paper,
      plot_bgcolor: c.paper,
      font: { family: "inherit", color: c.ink },
    };

    Plotly.newPlot(elementId, data, layout, { displayModeBar: false, responsive: true });
    chartRegistry.set(elementId, { type: "gauge", args: [value, threshold, maxDisplay] });
  }

  /* =========================================================
     4. PLOTLY SHAP BAR CHART
     ========================================================= */
  /**
   * Render a horizontal SHAP feature-contribution bar chart.
   * @param {string} elementId - container element id
   * @param {Array<{feature:string, value:number}>} shapData
   * @param {number} [topN=12] - max number of features to show
   */
  function renderShapChart(elementId, shapData, topN) {
    if (typeof Plotly === "undefined") {
      console.warn("Plotly.js is not loaded; SHAP chart skipped.");
      return;
    }
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!Array.isArray(shapData) || shapData.length === 0) {
      el.innerHTML = '<p class="text-muted" style="padding:12px 0;">No SHAP data available for this prediction.</p>';
      return;
    }

    const c = plotlyColors();
    const n = topN || 12;
    const sorted = [...shapData]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, n)
      .sort((a, b) => Math.abs(a.value) - Math.abs(b.value)); // ascending for horizontal bar display

    const data = [
      {
        type: "bar",
        orientation: "h",
        x: sorted.map((d) => d.value),
        y: sorted.map((d) => d.feature),
        marker: {
          color: sorted.map((d) => (d.value >= 0 ? "#dc3b3b" : "#0f9d6b")),
        },
        hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
      },
    ];

    const layout = {
      margin: { t: 10, b: 36, l: 150, r: 20 },
      paper_bgcolor: c.paper,
      plot_bgcolor: c.paper,
      font: { family: "inherit", color: c.ink, size: 12 },
      xaxis: {
        title: "SHAP value (impact on predicted risk)",
        zeroline: true,
        zerolinecolor: c.track,
        gridcolor: c.track,
        color: c.ink,
      },
      yaxis: { color: c.ink },
    };

    Plotly.newPlot(elementId, data, layout, { displayModeBar: false, responsive: true });
    chartRegistry.set(elementId, { type: "shap", args: [shapData, topN] });
  }

  /* Re-render all known charts with theme-appropriate colors */
  function refreshChartsForTheme() {
    chartRegistry.forEach((entry, elementId) => {
      if (entry.type === "gauge") renderGaugeChart(elementId, ...entry.args);
      if (entry.type === "shap") renderShapChart(elementId, ...entry.args);
    });
  }

  /* Auto-render charts declared via data attributes, e.g.:
     <div id="gauge-chart" data-gauge='{"value":0.23,"threshold":0.1183}'></div>
     <div id="shap-chart" data-shap='[{"feature":"Lactate","value":0.4}, ...]'></div>
  */
  function initDeclarativeCharts() {
    document.querySelectorAll("[data-gauge]").forEach((el) => {
      try {
        const cfg = JSON.parse(el.getAttribute("data-gauge"));
        renderGaugeChart(el.id, cfg.value, cfg.threshold, cfg.maxDisplay);
      } catch (e) {
        console.error("Invalid data-gauge JSON on #" + el.id, e);
      }
    });
    document.querySelectorAll("[data-shap]").forEach((el) => {
      try {
        const cfg = JSON.parse(el.getAttribute("data-shap"));
        renderShapChart(el.id, cfg, el.dataset.shapTop ? parseInt(el.dataset.shapTop, 10) : undefined);
      } catch (e) {
        console.error("Invalid data-shap JSON on #" + el.id, e);
      }
    });
  }

  /* =========================================================
     5. FORM VALIDATION
     - Add [data-validate] to a <form>
     - Add data-min / data-max / required to individual inputs
     - Known vital/lab fields get sane clinical-range defaults
       if min/max aren't explicitly set
     ========================================================= */
  const CLINICAL_RANGES = {
    HR: [0, 300], O2Sat: [0, 100], Temp: [25, 45], SBP: [0, 300], MAP: [0, 250],
    DBP: [0, 200], Resp: [0, 80], EtCO2: [0, 100], pH: [6.5, 8.0], PaCO2: [0, 150],
    BaseExcess: [-30, 30], HCO3: [0, 60], SaO2: [0, 100], FiO2: [0.21, 1],
    Glucose: [0, 1500], BUN: [0, 300], Creatinine: [0, 30], Calcium: [0, 20],
    Magnesium: [0, 10], Phosphate: [0, 20], Potassium: [0, 12], Lactate: [0, 30],
    Bilirubin_total: [0, 50], AST: [0, 5000], Alkalinephos: [0, 2000],
    TroponinI: [0, 100], Hct: [0, 75], Hgb: [0, 25], PTT: [0, 200], WBC: [0, 200],
    Fibrinogen: [0, 1500], Platelets: [0, 1500], Age: [0, 120],
  };

  function fieldErrorEl(input) {
    let err = input.parentElement.querySelector(".field-error-msg");
    if (!err) {
      err = document.createElement("span");
      err.className = "field-error-msg";
      err.style.color = "#dc3b3b";
      err.style.fontSize = "11.5px";
      err.style.marginTop = "2px";
      input.parentElement.appendChild(err);
    }
    return err;
  }

  function setFieldError(input, message) {
    input.style.borderColor = "#dc3b3b";
    input.style.boxShadow = "0 0 0 3px rgba(220,59,59,0.15)";
    input.setAttribute("aria-invalid", "true");
    const err = fieldErrorEl(input);
    err.textContent = message;
    err.style.display = "block";
  }

  function clearFieldError(input) {
    input.style.borderColor = "";
    input.style.boxShadow = "";
    input.removeAttribute("aria-invalid");
    const err = input.parentElement.querySelector(".field-error-msg");
    if (err) err.style.display = "none";
  }

  function validateField(input) {
    const name = input.name || input.id || "";
    const baseName = name.replace(/^p1_|^p2_|^h_/, "");
    const value = input.value.trim();

    // Required check
    if (input.hasAttribute("required") && value === "") {
      setFieldError(input, "This field is required.");
      return false;
    }

    // Empty + not required is always valid (blank labs/vitals are allowed)
    if (value === "") {
      clearFieldError(input);
      return true;
    }

    // Must be numeric for number inputs
    if (input.type === "number") {
      const n = parseFloat(value);
      if (Number.isNaN(n)) {
        setFieldError(input, "Enter a valid number.");
        return false;
      }

      const explicitMin = input.hasAttribute("data-min") ? parseFloat(input.dataset.min) : null;
      const explicitMax = input.hasAttribute("data-max") ? parseFloat(input.dataset.max) : null;
      const range = CLINICAL_RANGES[baseName];

      const min = explicitMin !== null ? explicitMin : (range ? range[0] : (input.min !== "" ? parseFloat(input.min) : null));
      const max = explicitMax !== null ? explicitMax : (range ? range[1] : (input.max !== "" ? parseFloat(input.max) : null));

      if (min !== null && n < min) {
        setFieldError(input, `Value seems too low (min ${min}).`);
        return false;
      }
      if (max !== null && n > max) {
        setFieldError(input, `Value seems too high (max ${max}).`);
        return false;
      }
    }

    clearFieldError(input);
    return true;
  }

  function validateForm(form) {
    const inputs = form.querySelectorAll("input, select, textarea");
    let valid = true;
    inputs.forEach((input) => {
      if (input.type === "hidden" || input.disabled) return;
      if (!validateField(input)) valid = false;
    });
    form.dataset.validated = valid ? "true" : "false";
    return valid;
  }

  function initFormValidation() {
    document.querySelectorAll("form[data-validate]").forEach((form) => {
      // live validation as the user types/leaves a field
      form.querySelectorAll("input, select, textarea").forEach((input) => {
        input.addEventListener("blur", () => validateField(input));
        input.addEventListener("input", () => {
          if (input.getAttribute("aria-invalid") === "true") validateField(input);
        });
      });

      form.addEventListener("submit", (e) => {
        const ok = validateForm(form);
        if (!ok) {
          e.preventDefault();
          const firstError = form.querySelector('[aria-invalid="true"]');
          if (firstError) {
            firstError.focus();
            firstError.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }
      });
    });
  }

  /* =========================================================
     INIT
     ========================================================= */
  function init() {
    initTheme();
    initAutoSpinners();
    initFormValidation();
    initDeclarativeCharts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  /* Public API */
  window.Dashboard = {
    showSpinner,
    hideSpinner,
    renderGaugeChart,
    renderShapChart,
    validateForm,
    validateField,
    applyTheme,
  };
})(window, document);
