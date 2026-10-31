// Tastatur/Maus mit Pointer-Lock. Fällt auf "Maus über dem Fenster" zurück,
// wenn der Browser Pointer-Lock verweigert.

export class Input {
  constructor(canvas) {
    this.canvas = canvas;
    this.keys = new Set();
    this.pressed = new Set(); // nur in diesem Frame gedrückt
    this.mouseDX = 0;
    this.mouseDY = 0;
    this.wheel = 0;
    this.lmb = false;
    this.rmb = false;
    this.lmbPressed = false;
    this.locked = false;
    this.enabled = false; // nur im Raid ohne offenes Menü
    this.sensitivity = 1;
    this.lockFailed = false;

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      const k = e.code;
      if (this.enabled && ['Tab', 'Space', 'AltLeft', 'AltRight', 'KeyQ'].includes(k)) e.preventDefault();
      this.keys.add(k);
      this.pressed.add(k);
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => {
      this.keys.clear();
      this.lmb = this.rmb = false;
    });

    canvas.addEventListener('mousedown', (e) => {
      if (!this.enabled) return;
      if (!this.locked && !this.lockFailed) this.requestLock();
      if (e.button === 0) { this.lmb = true; this.lmbPressed = true; }
      if (e.button === 2) this.rmb = true;
    });
    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.lmb = false;
      if (e.button === 2) this.rmb = false;
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('mousemove', (e) => {
      if (!this.enabled) return;
      if (this.locked || this.lockFailed) {
        this.mouseDX += e.movementX || 0;
        this.mouseDY += e.movementY || 0;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      if (this.enabled) this.wheel += Math.sign(e.deltaY);
    }, { passive: true });

    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === this.canvas;
      if (this.onLockChange) this.onLockChange(this.locked);
    });
    document.addEventListener('pointerlockerror', () => {
      this.lockFailed = true;
    });
  }

  requestLock() {
    try {
      const p = this.canvas.requestPointerLock();
      if (p && p.catch) p.catch(() => { this.lockFailed = true; });
    } catch {
      this.lockFailed = true;
    }
  }

  exitLock() {
    if (document.pointerLockElement) document.exitPointerLock();
  }

  down(code) { return this.keys.has(code); }
  hit(code) { return this.pressed.has(code); }

  endFrame() {
    this.pressed.clear();
    this.mouseDX = 0;
    this.mouseDY = 0;
    this.wheel = 0;
    this.lmbPressed = false;
  }

  reset() {
    this.keys.clear();
    this.pressed.clear();
    this.lmb = this.rmb = false;
    this.mouseDX = this.mouseDY = 0;
  }
}
