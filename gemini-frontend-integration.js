/**
 * Gemini AI Integration for MPS Connect Frontend
 *
 * This file provides JavaScript functions to interact with the new Gemini AI endpoints
 * for case analysis, letter generation, and approval recommendations.
 */

class GeminiIntegration {
  constructor(apiBase, apiKey) {
    this.apiBase = apiBase;
    this.apiKey = apiKey;
  }

  /**
   * Analyze case with Gemini Flash (preview)
   */
  async analyzeCasePreview(caseText) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/analyze-case-preview`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_text: caseText,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in case analysis preview:", error);
      throw error;
    }
  }

  /**
   * Analyze case with Gemini Pro (final)
   */
  async analyzeCaseFinal(caseText, feedback = null) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/analyze-case-final`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_text: caseText,
            feedback: feedback,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in case analysis final:", error);
      throw error;
    }
  }

  /**
   * Generate letter with Gemini Flash (preview)
   */
  async generateLetterPreview(caseData) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/generate-letter-preview`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_data: caseData,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in letter generation preview:", error);
      throw error;
    }
  }

  /**
   * Generate letter with Gemini Pro (final)
   */
  async generateLetterFinal(caseData, feedback = null) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/generate-letter-final`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_data: caseData,
            feedback: feedback,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in letter generation final:", error);
      throw error;
    }
  }

  /**
   * Get approval recommendation with Gemini Flash (preview)
   */
  async recommendApprovalPreview(caseAnalysis) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/recommend-approval-preview`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_analysis: caseAnalysis,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in approval recommendation preview:", error);
      throw error;
    }
  }

  /**
   * Get approval recommendation with Gemini Pro (final)
   */
  async recommendApprovalFinal(caseAnalysis, feedback = null) {
    try {
      const response = await fetch(
        `${this.apiBase}/api/gemini/recommend-approval-final`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": this.apiKey,
          },
          body: JSON.stringify({
            case_analysis: caseAnalysis,
            feedback: feedback,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error in approval recommendation final:", error);
      throw error;
    }
  }
}

/**
 * Enhanced UI Integration for Gemini Features
 */
class GeminiUI {
  constructor(geminiIntegration) {
    this.gemini = geminiIntegration;
    this.currentCaseData = null;
    this.currentAnalysis = null;
  }

  /**
   * Initialize Gemini UI elements
   */
  init() {
    this.addGeminiButtons();
    this.addGeminiModals();
  }

  /**
   * Add Gemini-powered buttons to existing UI
   */
  addGeminiButtons() {
    // Add to case analysis section
    const analysisSection = document.querySelector("#caseAnalysis");
    if (analysisSection) {
      const geminiAnalysisDiv = document.createElement("div");
      geminiAnalysisDiv.className = "gemini-analysis-section";
      geminiAnalysisDiv.innerHTML = `
                <h3>AI-Powered Analysis (Gemini)</h3>
                <button id="analyzeWithGeminiFlash" class="btn btn-primary">Quick Analysis (Flash)</button>
                <button id="analyzeWithGeminiPro" class="btn btn-success">Detailed Analysis (Pro)</button>
                <div id="geminiAnalysisResult" class="mt-3"></div>
            `;
      analysisSection.appendChild(geminiAnalysisDiv);

      // Add event listeners
      document
        .getElementById("analyzeWithGeminiFlash")
        .addEventListener("click", () => this.analyzeWithFlash());
      document
        .getElementById("analyzeWithGeminiPro")
        .addEventListener("click", () => this.analyzeWithPro());
    }

    // Add to letter generation section
    const letterSection = document.querySelector("#letterGeneration");
    if (letterSection) {
      const geminiLetterDiv = document.createElement("div");
      geminiLetterDiv.className = "gemini-letter-section";
      geminiLetterDiv.innerHTML = `
                <h3>AI-Powered Letter Generation (Gemini)</h3>
                <button id="generateLetterFlash" class="btn btn-primary">Quick Draft (Flash)</button>
                <button id="generateLetterPro" class="btn btn-success">Final Letter (Pro)</button>
                <div id="geminiLetterResult" class="mt-3"></div>
            `;
      letterSection.appendChild(geminiLetterDiv);

      // Add event listeners
      document
        .getElementById("generateLetterFlash")
        .addEventListener("click", () => this.generateLetterWithFlash());
      document
        .getElementById("generateLetterPro")
        .addEventListener("click", () => this.generateLetterWithPro());
    }
  }

  /**
   * Analyze case with Gemini Flash
   */
  async analyzeWithFlash() {
    const caseText = document.getElementById("caseText").value;
    if (!caseText.trim()) {
      alert("Please enter case text first");
      return;
    }

    try {
      const result = await this.gemini.analyzeCasePreview(caseText);
      this.displayAnalysisResult(result, "flash");
    } catch (error) {
      this.displayError("Analysis failed: " + error.message);
    }
  }

  /**
   * Analyze case with Gemini Pro
   */
  async analyzeWithPro() {
    const caseText = document.getElementById("caseText").value;
    if (!caseText.trim()) {
      alert("Please enter case text first");
      return;
    }

    const feedback = prompt(
      "Any additional context or feedback for the analysis?"
    );

    try {
      const result = await this.gemini.analyzeCaseFinal(caseText, feedback);
      this.displayAnalysisResult(result, "pro");
    } catch (error) {
      this.displayError("Analysis failed: " + error.message);
    }
  }

  /**
   * Generate letter with Gemini Flash
   */
  async generateLetterWithFlash() {
    if (!this.currentCaseData) {
      alert("Please analyze a case first");
      return;
    }

    try {
      const result = await this.gemini.generateLetterPreview(
        this.currentCaseData
      );
      this.displayLetterResult(result, "flash");
    } catch (error) {
      this.displayError("Letter generation failed: " + error.message);
    }
  }

  /**
   * Generate letter with Gemini Pro
   */
  async generateLetterWithPro() {
    if (!this.currentCaseData) {
      alert("Please analyze a case first");
      return;
    }

    const feedback = prompt(
      "Any specific requirements or feedback for the letter?"
    );

    try {
      const result = await this.gemini.generateLetterFinal(
        this.currentCaseData,
        feedback
      );
      this.displayLetterResult(result, "pro");
    } catch (error) {
      this.displayError("Letter generation failed: " + error.message);
    }
  }

  /**
   * Display analysis result
   */
  displayAnalysisResult(result, model) {
    const resultDiv = document.getElementById("geminiAnalysisResult");
    if (result.success) {
      const data = result.data;
      resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>Analysis Result (${model.toUpperCase()})</h4>
                    <p><strong>Priority:</strong> ${data.priority_level}</p>
                    <p><strong>Categories:</strong> ${data.categories
                      .map((c) => c.label)
                      .join(", ")}</p>
                    <p><strong>Key Facts:</strong></p>
                    <ul>${data.key_facts
                      .map((fact) => `<li>${fact}</li>`)
                      .join("")}</ul>
                    <p><strong>Reasoning:</strong> ${data.reasoning}</p>
                </div>
            `;
      this.currentAnalysis = data;
    } else {
      this.displayError(result.error || "Analysis failed");
    }
  }

  /**
   * Display letter result
   */
  displayLetterResult(result, model) {
    const resultDiv = document.getElementById("geminiLetterResult");
    if (result.success) {
      const data = result.data;
      resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>Letter Generated (${model.toUpperCase()})</h4>
                    <p><strong>Subject:</strong> ${data.subject}</p>
                    <p><strong>Tone:</strong> ${data.tone}</p>
                    <p><strong>Confidence:</strong> ${(
                      data.confidence * 100
                    ).toFixed(1)}%</p>
                    <div class="mt-3">
                        <h5>Letter Content:</h5>
                        <div class="border p-3" style="white-space: pre-wrap;">${
                          data.content
                        }</div>
                    </div>
                    ${
                      data.suggested_improvements.length > 0
                        ? `
                        <div class="mt-3">
                            <h5>Suggested Improvements:</h5>
                            <ul>${data.suggested_improvements
                              .map((improvement) => `<li>${improvement}</li>`)
                              .join("")}</ul>
                        </div>
                    `
                        : ""
                    }
                </div>
            `;
    } else {
      this.displayError(result.error || "Letter generation failed");
    }
  }

  /**
   * Display error message
   */
  displayError(message) {
    const resultDiv =
      document.getElementById("geminiAnalysisResult") ||
      document.getElementById("geminiLetterResult");
    if (resultDiv) {
      resultDiv.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${message}
                </div>
            `;
    }
  }

  /**
   * Add modals for detailed views
   */
  addGeminiModals() {
    // Implementation for modals if needed
  }
}

// Usage example:
// const gemini = new GeminiIntegration('https://your-api-url.com', 'your-api-key');
// const geminiUI = new GeminiUI(gemini);
// geminiUI.init();

