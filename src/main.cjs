const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const path = require('path');

function createMenu(win) {
  const template = [{
    label: 'View',
    submenu: [
      { role: 'reload' }, { role: 'forceReload' }, { type: 'separator' },
      { label: 'Toggle Developer Tools',
        accelerator: process.platform === 'darwin' ? 'Alt+Command+I' : 'Ctrl+Shift+I',
        click: () => win.webContents.toggleDevTools() }
    ]
  }];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200, height: 800,
    webPreferences: { contextIsolation: true, nodeIntegration: false, preload: path.join(__dirname, 'preload.cjs'), sandbox: true, devTools: true },
    title: 'PropFix Photo Editor'
  });
  createMenu(win);
  win.loadFile(path.join(__dirname, 'index.html'));
  win.webContents.openDevTools({ mode: 'detach' });
}

app.whenReady().then(() => {
  ipcMain.handle('chooseExportPath', async (_evt, suggested) => {
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'Export warped image',
      defaultPath: suggested || 'output.png',
      filters: [{ name: 'PNG Image', extensions: ['png'] }]
    });
    return canceled ? null : filePath;
  });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });


