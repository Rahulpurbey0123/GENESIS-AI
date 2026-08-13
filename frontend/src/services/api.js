/**
 * Centralized API Service Layer for GENESIS-AI Dashboard.
 * Encapsulates all backend HTTP interactions with typed responses and error handling.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function handleResponse(response) {
  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
      }
    } catch (e) {
      // Use fallback errorMsg
    }
    throw new Error(errorMsg);
  }
  return await response.json();
}

export const apiService = {
  /**
   * Upload CSV dataset
   */
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE_URL}/api/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  },

  /**
   * Get metadata for uploaded dataset
   */
  async getDataset(datasetId) {
    const res = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}`);
    return await handleResponse(res);
  },

  /**
   * Generate Dataset Intelligence Profile (DIP) v1.1
   */
  async profileDataset(datasetId, targetColumn) {
    const formData = new FormData();
    formData.append('target_column', targetColumn);
    const res = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/profile`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  },

  /**
   * Get stored DIP profile
   */
  async getDatasetProfile(datasetId, targetColumn) {
    const url = targetColumn
      ? `${API_BASE_URL}/api/datasets/${datasetId}/profile?target_column=${encodeURIComponent(targetColumn)}`
      : `${API_BASE_URL}/api/datasets/${datasetId}/profile`;
    const res = await fetch(url);
    return await handleResponse(res);
  },

  /**
   * Get LLM configuration status
   */
  async getLLMConfig() {
    const res = await fetch(`${API_BASE_URL}/api/config/llm`);
    return await handleResponse(res);
  },

  /**
   * Get LLM provider status
   */
  async getLLMStatus() {
    const res = await fetch(`${API_BASE_URL}/api/llm/status`);
    return await handleResponse(res);
  },

  /**
   * Get read-only recommendations for dataset WITHOUT starting an experiment
   */
  async getDatasetRecommendations(datasetId, targetColumn) {
    const formData = new FormData();
    if (targetColumn) formData.append('target_column', targetColumn);
    const res = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/recommendations`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  },

  /**
   * Create & launch optimization experiment
   */
  async createExperiment({ datasetId, targetColumn, mode = 'genesis', topK = 2, populationSize = 20, generations = 10, maxEvaluations = 200 }) {
    const formData = new FormData();
    formData.append('dataset_id', datasetId);
    formData.append('target_column', targetColumn);
    formData.append('mode', mode);
    formData.append('top_k', topK);
    formData.append('population_size', populationSize);
    formData.append('generations', generations);
    formData.append('max_evaluations', maxEvaluations);

    const res = await fetch(`${API_BASE_URL}/api/experiments`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  },

  /**
   * List all stored experiments
   */
  async listExperiments() {
    const res = await fetch(`${API_BASE_URL}/api/experiments`);
    return await handleResponse(res);
  },

  /**
   * Get real-time status of an experiment
   */
  async getExperiment(experimentId) {
    const res = await fetch(`${API_BASE_URL}/api/experiments/${experimentId}`);
    return await handleResponse(res);
  },

  /**
   * Get recommendations and candidate search space for an experiment
   */
  async getRecommendations(experimentId) {
    const res = await fetch(`${API_BASE_URL}/api/experiments/${experimentId}/recommendations`);
    return await handleResponse(res);
  },

  /**
   * Get evaluation results and best pipeline metrics
   */
  async getResults(experimentId) {
    const res = await fetch(`${API_BASE_URL}/api/experiments/${experimentId}/results`);
    return await handleResponse(res);
  },

  /**
   * Get SHAP and explainability outputs
   */
  async getExplanations(experimentId) {
    const res = await fetch(`${API_BASE_URL}/api/experiments/${experimentId}/explanations`);
    return await handleResponse(res);
  },

  /**
   * Send question to evidence-grounded AI Assistant
   */
  async sendChatMessage(experimentId, prompt) {
    const formData = new FormData();
    formData.append('prompt', prompt);
    const res = await fetch(`${API_BASE_URL}/api/experiments/${experimentId}/chat`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse(res);
  }
};
