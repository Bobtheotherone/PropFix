const { contextBridge, ipcRenderer } = require('electron');

// Expose APIs to the renderer through the `window.electron` namespace.
contextBridge.exposeInMainWorld('electron', {
  /**
   * Open a file dialog and return selected file paths.
   * @returns {Promise<string[]>} list of selected file paths
   */
  openFiles: async () => {
    return ipcRenderer.invoke('dialog:openFile');
  },

  /**
   * Send an AI processing request to the main process.  The command
   * parameter should be a JSON‑serialisable object describing the
   * operation (e.g., {command: 'enhance', params: {...}}).
   * @param {object} command
   * @returns {Promise<string>} result from the Python back‑end
   */
  processImage: async (command) => {
    return ipcRenderer.invoke('ai:process', command);
  }
});
