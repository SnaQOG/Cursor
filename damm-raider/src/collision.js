// Statische Kollision: achsenparallele Boxen + senkrechte Zylinder in einem
// XZ-Raster, plus Terrain-Raycast.
import { clamp } from './util.js';

const CELL = 16;

export class CollisionWorld {
  constructor(terrain) {
    this.terrain = terrain;
    this.half = terrain.half;
    this.dim = Math.ceil((this.half * 2) / CELL);
    this.cells = new Array(this.dim * this.dim);
    for (let i = 0; i < this.cells.length; i++) this.cells[i] = [];
    this.shapes = [];
    this.stamp = 1;
    this._tmp = [];
  }

  cellIndex(ix, iz) { return iz * this.dim + ix; }
  toCell(v) { return clamp(Math.floor((v + this.half) / CELL), 0, this.dim - 1); }

  insert(s) {
    s.id = this.shapes.length;
    s.active = true;
    s.mark = 0;
    this.shapes.push(s);
    const x0 = this.toCell(s.minX), x1 = this.toCell(s.maxX);
    const z0 = this.toCell(s.minZ), z1 = this.toCell(s.maxZ);
    for (let iz = z0; iz <= z1; iz++) for (let ix = x0; ix <= x1; ix++) this.cells[this.cellIndex(ix, iz)].push(s);
    return s;
  }

  addBox(minX, minY, minZ, maxX, maxY, maxZ, tag = null) {
    return this.insert({ kind: 'box', minX, minY, minZ, maxX, maxY, maxZ, tag });
  }

  addCyl(x, z, r, minY, maxY, tag = null) {
    return this.insert({ kind: 'cyl', x, z, r, minX: x - r, maxX: x + r, minZ: z - r, maxZ: z + r, minY, maxY, tag });
  }

  query(minX, minZ, maxX, maxZ) {
    const out = this._tmp;
    out.length = 0;
    const st = ++this.stamp;
    const x0 = this.toCell(minX), x1 = this.toCell(maxX);
    const z0 = this.toCell(minZ), z1 = this.toCell(maxZ);
    for (let iz = z0; iz <= z1; iz++) {
      for (let ix = x0; ix <= x1; ix++) {
        const cell = this.cells[this.cellIndex(ix, iz)];
        for (let k = 0; k < cell.length; k++) {
          const s = cell[k];
          if (s.mark === st || !s.active) continue;
          s.mark = st;
          if (s.maxX < minX || s.minX > maxX || s.maxZ < minZ || s.minZ > maxZ) continue;
          out.push(s);
        }
      }
    }
    return out;
  }

  // Höchste begehbare Fläche unter einem Kreis (Boxen-Oberseiten + Terrain)
  groundAt(x, z, r, feetY, stepUp) {
    let g = this.terrain.height(x, z);
    const list = this.query(x - r, z - r, x + r, z + r);
    for (const s of list) {
      if (s.maxY > feetY + stepUp) continue;
      if (s.maxY <= g) continue;
      if (s.kind === 'box') {
        const cx = clamp(x, s.minX, s.maxX), cz = clamp(z, s.minZ, s.maxZ);
        const dx = x - cx, dz = z - cz;
        if (dx * dx + dz * dz < r * r * 0.5) g = s.maxY;
      } else {
        const d = Math.hypot(x - s.x, z - s.z);
        if (d < s.r + r * 0.5) g = s.maxY;
      }
    }
    return g;
  }

  // Decke über dem Kopf (für Sprünge)
  ceilingAt(x, z, r, fromY, toY) {
    let c = Infinity;
    const list = this.query(x - r, z - r, x + r, z + r);
    for (const s of list) {
      if (s.minY < fromY || s.minY > toY) continue;
      if (s.kind === 'box') {
        const cx = clamp(x, s.minX, s.maxX), cz = clamp(z, s.minZ, s.maxZ);
        if ((x - cx) ** 2 + (z - cz) ** 2 < r * r * 0.5) c = Math.min(c, s.minY);
      } else if (Math.hypot(x - s.x, z - s.z) < s.r) c = Math.min(c, s.minY);
    }
    return c;
  }

  // Schiebt einen senkrechten Zylinder (Füße bei y) aus allen Hindernissen heraus.
  // pos wird verändert. Gibt true zurück, wenn es eine Kollision gab.
  pushOut(pos, r, height, stepUp) {
    let hit = false;
    for (let iter = 0; iter < 3; iter++) {
      const list = this.query(pos.x - r, pos.z - r, pos.x + r, pos.z + r);
      let moved = false;
      for (const s of list) {
        if (s.maxY <= pos.y + stepUp || s.minY >= pos.y + height) continue;
        if (s.kind === 'box') {
          const cx = clamp(pos.x, s.minX, s.maxX), cz = clamp(pos.z, s.minZ, s.maxZ);
          let dx = pos.x - cx, dz = pos.z - cz;
          const d2 = dx * dx + dz * dz;
          if (d2 >= r * r) continue;
          if (d2 > 1e-8) {
            const d = Math.sqrt(d2);
            pos.x += (dx / d) * (r - d);
            pos.z += (dz / d) * (r - d);
          } else {
            // Mittelpunkt in der Box: kürzester Ausweg
            const pl = pos.x - s.minX, pr = s.maxX - pos.x, pb = pos.z - s.minZ, pf = s.maxZ - pos.z;
            const m = Math.min(pl, pr, pb, pf);
            if (m === pl) pos.x = s.minX - r;
            else if (m === pr) pos.x = s.maxX + r;
            else if (m === pb) pos.z = s.minZ - r;
            else pos.z = s.maxZ + r;
          }
          moved = hit = true;
        } else {
          const dx = pos.x - s.x, dz = pos.z - s.z;
          const d = Math.sqrt(dx * dx + dz * dz);
          const min = s.r + r;
          if (d >= min) continue;
          if (d > 1e-6) {
            pos.x = s.x + (dx / d) * min;
            pos.z = s.z + (dz / d) * min;
          } else pos.x = s.x + min;
          moved = hit = true;
        }
      }
      if (!moved) break;
    }
    return hit;
  }

  // Punkt in einem Hindernis?
  pointBlocked(x, y, z) {
    const list = this.query(x, z, x, z);
    for (const s of list) {
      if (y < s.minY || y > s.maxY) continue;
      if (s.kind === 'box') {
        if (x >= s.minX && x <= s.maxX && z >= s.minZ && z <= s.maxZ) return true;
      } else if (Math.hypot(x - s.x, z - s.z) <= s.r) return true;
    }
    return y < this.terrain.height(x, z);
  }

  // Strahl gegen Boxen/Zylinder. Liefert {t, nx, ny, nz, shape} oder null.
  raycastShapes(ox, oy, oz, dx, dy, dz, maxT) {
    // 2D-DDA durch das Raster
    const st = ++this.stamp;
    let best = null;
    let bestT = maxT;
    const len2 = Math.hypot(dx, dz);
    let ix = this.toCell(ox), iz = this.toCell(oz);
    const stepX = dx > 0 ? 1 : -1, stepZ = dz > 0 ? 1 : -1;
    const cellMinX = (i) => -this.half + i * CELL;
    let tMaxX = Math.abs(dx) < 1e-9 ? Infinity : ((dx > 0 ? cellMinX(ix + 1) : cellMinX(ix)) - ox) / dx;
    let tMaxZ = Math.abs(dz) < 1e-9 ? Infinity : ((dz > 0 ? cellMinX(iz + 1) : cellMinX(iz)) - oz) / dz;
    const tDX = Math.abs(dx) < 1e-9 ? Infinity : CELL / Math.abs(dx);
    const tDZ = Math.abs(dz) < 1e-9 ? Infinity : CELL / Math.abs(dz);
    let tEnter = 0;
    const res = { t: 0, nx: 0, ny: 0, nz: 0 };
    for (let guard = 0; guard < 200; guard++) {
      if (ix < 0 || iz < 0 || ix >= this.dim || iz >= this.dim) break;
      const cell = this.cells[this.cellIndex(ix, iz)];
      for (let k = 0; k < cell.length; k++) {
        const s = cell[k];
        if (s.mark === st || !s.active) continue;
        s.mark = st;
        const ok = s.kind === 'box'
          ? rayBox(ox, oy, oz, dx, dy, dz, s, bestT, res)
          : rayCyl(ox, oy, oz, dx, dy, dz, s, bestT, res);
        if (ok) {
          bestT = res.t;
          best = { t: res.t, nx: res.nx, ny: res.ny, nz: res.nz, shape: s };
        }
      }
      if (len2 < 1e-9) break; // senkrechter Strahl: nur eine Zelle
      if (tMaxX < tMaxZ) { tEnter = tMaxX; tMaxX += tDX; ix += stepX; }
      else { tEnter = tMaxZ; tMaxZ += tDZ; iz += stepZ; }
      if (tEnter > bestT) break;
    }
    return best;
  }

  raycastTerrain(ox, oy, oz, dx, dy, dz, maxT, stepLen = 1) {
    const T = this.terrain;
    let prevT = 0;
    let prevAbove = oy - T.height(ox, oz);
    if (prevAbove < 0) return { t: 0 };
    for (let t = stepLen; t <= maxT + stepLen; t += stepLen) {
      const tt = Math.min(t, maxT);
      const x = ox + dx * tt, y = oy + dy * tt, z = oz + dz * tt;
      const above = y - T.height(x, z);
      if (above < 0) {
        // Binärsuche
        let a = prevT, b = tt;
        for (let i = 0; i < 8; i++) {
          const m = (a + b) * 0.5;
          const ym = oy + dy * m - T.height(ox + dx * m, oz + dz * m);
          if (ym < 0) b = m; else a = m;
        }
        return { t: b };
      }
      prevT = tt;
      prevAbove = above;
      if (tt >= maxT) break;
      // schneller voran, wenn weit über dem Boden
      if (above > 12) t += stepLen * 3;
    }
    return null;
  }

  // Kombinierter Raycast. dir muss normiert sein.
  raycast(o, d, maxT, terrainStep = 1) {
    const s = this.raycastShapes(o.x, o.y, o.z, d.x, d.y, d.z, maxT);
    const lim = s ? s.t : maxT;
    const tr = this.raycastTerrain(o.x, o.y, o.z, d.x, d.y, d.z, lim, terrainStep);
    if (tr && (!s || tr.t < s.t)) {
      const x = o.x + d.x * tr.t, z = o.z + d.z * tr.t;
      const e = 0.5;
      const T = this.terrain;
      const nx = T.height(x - e, z) - T.height(x + e, z);
      const nz = T.height(x, z - e) - T.height(x, z + e);
      const l = Math.hypot(nx, 2 * e, nz);
      return { t: tr.t, nx: nx / l, ny: (2 * e) / l, nz: nz / l, shape: null, terrain: true };
    }
    return s;
  }

  // Sichtlinie frei?
  lineOfSight(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 0.01) return true;
    const d = { x: dx / len, y: dy / len, z: dz / len };
    const hit = this.raycast(a, d, len - 0.3, 2);
    return !hit;
  }
}

function rayBox(ox, oy, oz, dx, dy, dz, s, maxT, res) {
  let tmin = 0, tmax = maxT, nAxis = -1, nSign = 0;
  // X
  if (Math.abs(dx) < 1e-9) {
    if (ox < s.minX || ox > s.maxX) return false;
  } else {
    const inv = 1 / dx;
    let t1 = (s.minX - ox) * inv, t2 = (s.maxX - ox) * inv, sg = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; sg = 1; }
    if (t1 > tmin) { tmin = t1; nAxis = 0; nSign = sg; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  // Y
  if (Math.abs(dy) < 1e-9) {
    if (oy < s.minY || oy > s.maxY) return false;
  } else {
    const inv = 1 / dy;
    let t1 = (s.minY - oy) * inv, t2 = (s.maxY - oy) * inv, sg = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; sg = 1; }
    if (t1 > tmin) { tmin = t1; nAxis = 1; nSign = sg; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  // Z
  if (Math.abs(dz) < 1e-9) {
    if (oz < s.minZ || oz > s.maxZ) return false;
  } else {
    const inv = 1 / dz;
    let t1 = (s.minZ - oz) * inv, t2 = (s.maxZ - oz) * inv, sg = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; sg = 1; }
    if (t1 > tmin) { tmin = t1; nAxis = 2; nSign = sg; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  if (nAxis < 0) return false; // Start innerhalb der Box – ignorieren
  res.t = tmin;
  res.nx = nAxis === 0 ? nSign : 0;
  res.ny = nAxis === 1 ? nSign : 0;
  res.nz = nAxis === 2 ? nSign : 0;
  return true;
}

function rayCyl(ox, oy, oz, dx, dy, dz, s, maxT, res) {
  const px = ox - s.x, pz = oz - s.z;
  const a = dx * dx + dz * dz;
  let t = Infinity, nx = 0, ny = 0, nz = 0;
  if (a > 1e-9) {
    const b = 2 * (px * dx + pz * dz);
    const c = px * px + pz * pz - s.r * s.r;
    const disc = b * b - 4 * a * c;
    if (disc >= 0) {
      const t0 = (-b - Math.sqrt(disc)) / (2 * a);
      if (t0 > 0) {
        const y = oy + dy * t0;
        if (y >= s.minY && y <= s.maxY) {
          t = t0;
          nx = (px + dx * t0) / s.r; nz = (pz + dz * t0) / s.r;
        }
      }
    }
  }
  // Deckel oben
  if (Math.abs(dy) > 1e-9 && oy > s.maxY && dy < 0) {
    const tc = (s.maxY - oy) / dy;
    const x = px + dx * tc, z = pz + dz * tc;
    if (tc > 0 && tc < t && x * x + z * z <= s.r * s.r) { t = tc; nx = 0; ny = 1; nz = 0; }
  }
  if (t < maxT) {
    res.t = t; res.nx = nx; res.ny = ny; res.nz = nz;
    return true;
  }
  return false;
}
