// Gegnerische Geschosse, Raketen und Granaten.
import * as THREE from 'three';
import { rand } from './util.js';

const _d = new THREE.Vector3();
const _p = new THREE.Vector3();
const _q = new THREE.Vector3();
const _n = new THREE.Vector3();

function segPointDist(a, b, p) {
  _q.subVectors(b, a);
  const l2 = _q.lengthSq();
  let t = l2 > 0 ? _p.subVectors(p, a).dot(_q) / l2 : 0;
  t = Math.max(0, Math.min(1, t));
  _p.copy(a).addScaledVector(_q, t);
  return { d: _p.distanceTo(p), t };
}

export class Projectiles {
  constructor(game) {
    this.game = game;
    this.list = [];
    const add = (color) => new THREE.MeshBasicMaterial({ color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.geo = {
      bullet: new THREE.BoxGeometry(0.07, 0.07, 0.9),
      heavy: new THREE.BoxGeometry(0.09, 0.09, 1.2),
      bolt: new THREE.SphereGeometry(0.18, 8, 6),
      rocket: new THREE.CylinderGeometry(0.12, 0.12, 0.8, 8).rotateX(Math.PI / 2),
      grenade: new THREE.SphereGeometry(0.11, 10, 8),
    };
    this.mat = {
      bullet: add(0xff7a3a),
      heavy: add(0xffb040),
      bolt: add(0x6fd8ff),
      rocket: new THREE.MeshStandardMaterial({ color: 0x3a3d40, emissive: 0xff5010, emissiveIntensity: 0.4 }),
      grenade: new THREE.MeshStandardMaterial({ color: 0x4f5a3a, roughness: 0.6 }),
    };
    this.pool = { bullet: [], heavy: [], bolt: [], rocket: [], grenade: [] };
  }

  mesh(kind) {
    const m = this.pool[kind].pop() || new THREE.Mesh(this.geo[kind], this.mat[kind]);
    m.visible = true;
    if (!m.parent) this.game.scene.add(m);
    return m;
  }

  release(p) {
    p.mesh.visible = false;
    this.pool[p.kind].push(p.mesh);
  }

  enemyShot(pos, vel, dmg, kind, owner) {
    this.list.push({ kind, pos, vel, dmg, owner, life: 3, mesh: this.mesh(kind), enemy: true });
  }

  rocket(pos, vel, owner) {
    this.list.push({ kind: 'rocket', pos: pos.clone(), vel: vel.clone(), dmg: 28, owner, life: 6, mesh: this.mesh('rocket'), enemy: true, arm: 0.15 });
  }

  grenade(pos, vel, owner) {
    this.list.push({ kind: 'grenade', pos: pos.clone(), vel: vel.clone(), owner, life: 2.6, mesh: this.mesh('grenade'), enemy: false, bounces: 0 });
  }

  update(dt) {
    const game = this.game;
    const player = game.player;
    const chest = player.chest(new THREE.Vector3());
    const head = new THREE.Vector3(player.pos.x, player.pos.y + (player.crouch ? 1.15 : 1.6), player.pos.z);
    for (let i = this.list.length - 1; i >= 0; i--) {
      const p = this.list[i];
      p.life -= dt;
      const prev = p.pos.clone();
      if (p.kind === 'rocket') {
        p.arm -= dt;
        // leichtes Zielsuchen
        if (player.alive) {
          _d.subVectors(chest, p.pos).normalize();
          const sp = p.vel.length();
          _n.copy(p.vel).divideScalar(sp).lerp(_d, Math.min(1, dt * 0.9)).normalize();
          p.vel.copy(_n).multiplyScalar(sp);
        }
        if (Math.random() < 0.9) game.effects.smokePuff(p.pos, false);
        game.effects.fire(p.pos);
      } else if (p.kind === 'grenade') {
        p.vel.y -= 16 * dt;
      } else if (p.kind === 'bolt' && Math.random() < 0.5) {
        game.effects.thrusterTrail(p.pos, [0.4, 0.85, 1]);
      }
      p.pos.addScaledVector(p.vel, dt);

      const seg = _d.subVectors(p.pos, prev);
      const len = seg.length();
      let worldHit = null;
      if (len > 0.0001) {
        seg.divideScalar(len);
        worldHit = game.col.raycast(prev, seg.clone(), len, 0.5);
      }
      let removed = false;
      // Treffer auf den Spieler
      if (p.enemy && player.alive) {
        const r1 = segPointDist(prev, p.pos, chest);
        const r2 = segPointDist(prev, p.pos, head);
        const rad = p.kind === 'rocket' ? 0.8 : 0.5;
        const hitT = r1.d < rad ? r1.t : r2.d < 0.3 ? r2.t : -1;
        if (hitT >= 0 && (!worldHit || hitT * len < worldHit.t)) {
          if (p.kind === 'rocket') this.explodeRocket(p, p.pos);
          else {
            player.damage(p.dmg, p.owner ? p.owner.pos : prev, p.kind);
            game.effects.sparks(chest, _n.set(0, 1, 0), 5, [1, 0.5, 0.3], 3);
          }
          removed = true;
        }
      }
      if (!removed && worldHit) {
        const hp = prev.clone().addScaledVector(seg, worldHit.t);
        _n.set(worldHit.nx, worldHit.ny, worldHit.nz);
        if (p.kind === 'grenade') {
          const vn = p.vel.dot(_n);
          p.vel.addScaledVector(_n, -1.6 * vn).multiplyScalar(0.45);
          p.pos.copy(hp).addScaledVector(_n, 0.12);
          if (Math.abs(vn) > 2) game.sfx.play('grenadeBounce', hp);
        } else if (p.kind === 'rocket') {
          if (worldHit.shape && worldHit.shape.tag && worldHit.shape.tag.barrel) worldHit.shape.tag.barrel.explode();
          this.explodeRocket(p, hp);
          removed = true;
        } else {
          game.effects.sparks(hp, _n, 4, p.kind === 'bolt' ? [0.5, 0.85, 1] : [1, 0.6, 0.3], 4);
          if (worldHit.terrain) game.effects.dust(hp, _n, undefined, 2);
          removed = true;
        }
      }
      // Wasser
      if (!removed && p.kind !== 'grenade') {
        const w = game.terrain.waterAt(p.pos.x, p.pos.z);
        if (w && p.pos.y < w.level && prev.y >= w.level) {
          game.effects.splash(_p.set(p.pos.x, w.level, p.pos.z), w.toxic);
          if (p.kind === 'rocket') this.explodeRocket(p, p.pos);
          removed = true;
        }
      }
      if (!removed && p.life <= 0) {
        if (p.kind === 'grenade') game.explode(p.pos.clone(), 7.5, 95, p.owner);
        else if (p.kind === 'rocket') this.explodeRocket(p, p.pos);
        removed = true;
      }
      if (removed) {
        this.release(p);
        this.list.splice(i, 1);
        continue;
      }
      p.mesh.position.copy(p.pos);
      if (p.kind !== 'grenade' && p.kind !== 'bolt') p.mesh.lookAt(_p.copy(p.pos).add(p.vel));
      else if (p.kind === 'grenade') p.mesh.rotation.x += dt * 10;
    }
  }

  explodeRocket(p, pos) {
    this.game.explode(pos.clone(), 4.5, p.dmg, p.owner);
  }

  clear() {
    for (const p of this.list) this.release(p);
    this.list.length = 0;
  }
}

export function randomCone(dir, deg) {
  const s = (deg * Math.PI) / 180;
  return new THREE.Vector3(dir.x + rand(-s, s), dir.y + rand(-s, s), dir.z + rand(-s, s)).normalize();
}
