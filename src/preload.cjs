const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('env', { serverURL: 'http://127.0.0.1:5001' });
contextBridge.exposeInMainWorld('api', { chooseExportPath: (suggested) => ipcRenderer.invoke('chooseExportPath', suggested) });
