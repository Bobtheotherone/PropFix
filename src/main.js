const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

/**
 * Creates the main application window.  The window loads the renderer
 * (index.html) and sets up IPC handlers for communication with the
 * Python backâ€‘end.  Electronâ€™s `contextIsolation` and `preload` options
 * are enabled to follow best practices for security.
 */
function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile(path.join(__dirname, 'index.html'));
}

// Called when Electron has finished initialisation
app.whenReady().then(() => {
  createWindow();

  // On macOS it is common to recreate a window when the dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit the app when all windows are closed except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

/**
 * IPC handler: open a file dialog and return the selected paths.
 */
ipcMain.handle('dialog:openFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'] }
    ]
  });
  if (canceled) {
    return [];
  }
  return filePaths;
});

/**
 * IPC handler: perform an AI operation by sending a command to the
 * Python backâ€‘end.  The handler spawns a Python process using
 * `python-shell` and returns the result to the renderer.
 */
ipcMain.handle('ai:process', async (_event, command) => {
  return new Promise((resolve, reject) => {
    const pyshell = new PythonShell(require('path').join(__dirname, '..', 'python', 'image_processing.py'));
    let resultData = '';
    pyshell.on('message', (message) => {
      resultData += message;
    });
    pyshell.send(JSON.stringify(command));
    pyshell.end((err) => {
      if (err) {
        reject(err);
      } else {
        resolve(resultData);
      }
    });
  });
});
